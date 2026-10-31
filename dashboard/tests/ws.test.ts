import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  backoffDelay,
  type ConnectionStatus,
  LiveClient,
  liveUrl,
  type SocketFactory,
  type SocketHandlers,
} from '../src/lib/ws.ts';

/** A scriptable stand-in for a WebSocket: the test plays the server. */
class FakeSocket {
  closed: { code?: number; reason?: string } | null = null;
  readonly url: string;
  private readonly handlers: SocketHandlers;

  constructor(url: string, handlers: SocketHandlers) {
    this.url = url;
    this.handlers = handlers;
  }

  open(): void {
    this.handlers.onOpen();
  }

  send(type: string, data: unknown): void {
    this.handlers.onMessage(JSON.stringify({ type, data }));
  }

  sendRaw(raw: unknown): void {
    this.handlers.onMessage(raw);
  }

  /** Server-side or network close. */
  drop(code = 1006): void {
    this.handlers.onError();
    this.handlers.onClose(code, '');
  }

  close(code?: number, reason?: string): void {
    this.closed = { code, reason };
  }
}

function harness(options: { staleAfterMs?: number; jitter?: number; random?: () => number } = {}) {
  const sockets: FakeSocket[] = [];
  const factory: SocketFactory = (url, handlers) => {
    const socket = new FakeSocket(url, handlers);
    sockets.push(socket);
    return socket;
  };
  const rejected: string[] = [];
  const client = new LiveClient({
    url: 'ws://test/ws/live',
    socketFactory: factory,
    backoff: { initialMs: 100, maxMs: 1_000, jitter: options.jitter ?? 0 },
    staleAfterMs: options.staleAfterMs ?? 5_000,
    random: options.random ?? (() => 0.5),
    onRejected: (detail) => rejected.push(detail),
  });
  const statuses: ConnectionStatus[] = [];
  client.onStatus((s) => statuses.push(s));
  const last = () => {
    const socket = sockets.at(-1);
    if (!socket) throw new Error('no socket opened yet');
    return socket;
  };
  return { client, sockets, last, statuses, rejected };
}

const bits = { t: 1, bits: '0101' };

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe('LiveClient', () => {
  it('connects on start and reports connecting -> open', () => {
    const { client, sockets, last, statuses } = harness();
    client.start();
    expect(sockets).toHaveLength(1);
    expect(last().url).toBe('ws://test/ws/live');
    last().open();
    expect(statuses.map((s) => s.state)).toEqual(['idle', 'connecting', 'open']);
    client.start(); // idempotent while running
    expect(sockets).toHaveLength(1);
  });

  it('delivers decoded data to typed subscribers and caches the latest value', () => {
    const { client, last } = harness();
    const received: string[] = [];
    const off = client.on('bits', (data) => received.push(data.bits));
    client.start();
    last().open();
    last().send('bits', bits);
    expect(received).toEqual(['0101']);
    expect(client.latest('bits')).toEqual(bits);
    expect(client.latest('frame')).toBeUndefined();
    off();
    last().send('bits', { t: 2, bits: '1111' });
    expect(received).toEqual(['0101']);
    expect(client.latest('bits')?.bits).toBe('1111');
  });

  it('rejects invalid messages without delivering them, and ignores unknown types', () => {
    const { client, last, rejected } = harness();
    const handler = vi.fn();
    client.on('bits', handler);
    client.start();
    last().open();
    last().send('bits', { t: 'later', bits: '01' });
    last().sendRaw('not json');
    last().send('future_feature', { x: 1 });
    expect(handler).not.toHaveBeenCalled();
    expect(client.status.rejected).toBe(2);
    expect(rejected).toHaveLength(2);
    expect(rejected[0]).toMatch(/^invalid_data: data\.t/);
  });

  it('isolates subscribers: one throwing handler does not starve the others', () => {
    const { client, last } = harness();
    const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    const healthy = vi.fn();
    client.on('bits', () => {
      throw new Error('panel bug');
    });
    client.on('bits', healthy);
    client.start();
    last().open();
    last().send('bits', bits);
    expect(healthy).toHaveBeenCalledOnce();
    expect(errors).toHaveBeenCalledOnce();
  });

  it('reconnects with exponential backoff capped at maxMs', () => {
    const { client, sockets, last, statuses } = harness();
    client.start();
    const delays: number[] = [];
    for (let i = 0; i < 6; i++) {
      last().drop();
      const status = statuses.at(-1);
      expect(status?.state).toBe('reconnecting');
      delays.push(status?.retryInMs ?? -1);
      vi.advanceTimersByTime(status?.retryInMs ?? 0);
    }
    expect(delays).toEqual([100, 200, 400, 800, 1_000, 1_000]);
    expect(sockets).toHaveLength(7);
    expect(client.status.state).toBe('connecting');
  });

  it('waits the full delay before reconnecting', () => {
    const { client, sockets, last } = harness();
    client.start();
    last().drop();
    vi.advanceTimersByTime(99);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(sockets).toHaveLength(2);
  });

  it('resets the backoff only after a valid message, not on open alone', () => {
    const { client, last } = harness();
    client.start();
    last().drop();
    vi.advanceTimersByTime(100);
    // A flapping server: accepts, then drops before sending anything.
    last().open();
    last().drop();
    expect(client.status.attempt).toBe(2);
    expect(client.status.retryInMs).toBe(200);
    vi.advanceTimersByTime(200);
    last().open();
    last().send('bits', bits);
    expect(client.status.attempt).toBe(0);
    last().drop();
    expect(client.status.retryInMs).toBe(100);
  });

  it('stop() cancels a pending reconnect and closes the socket', () => {
    const { client, sockets, last } = harness();
    client.start();
    last().drop();
    client.stop();
    vi.advanceTimersByTime(10_000);
    expect(sockets).toHaveLength(1);
    expect(client.status.state).toBe('stopped');

    client.start();
    last().open();
    client.stop();
    expect(last().closed).toEqual({ code: 1001, reason: 'client stopped' });
  });

  it('ignores late events from a socket it has abandoned', () => {
    const { client, sockets, last } = harness();
    const handler = vi.fn();
    client.on('bits', handler);
    client.start();
    const first = last();
    first.drop();
    vi.advanceTimersByTime(100);
    const second = last();
    second.open();
    first.send('bits', bits); // zombie socket
    first.drop();
    expect(handler).not.toHaveBeenCalled();
    expect(client.status.state).toBe('open');
    expect(sockets).toHaveLength(2);
  });

  it('closes and replaces a connection that goes silent', () => {
    const { client, sockets, last } = harness({ staleAfterMs: 4_000 });
    client.start();
    last().open();
    vi.advanceTimersByTime(3_000);
    last().send('bits', bits); // still alive
    vi.advanceTimersByTime(3_000);
    expect(sockets).toHaveLength(1);
    vi.advanceTimersByTime(1_000); // 4 s of silence: the watchdog's next tick fires
    const stale = sockets[0];
    expect(stale?.closed).toEqual({ code: 1001, reason: 'no data' });
    expect(client.status.state).toBe('reconnecting');
    vi.advanceTimersByTime(client.status.retryInMs ?? 0);
    expect(sockets).toHaveLength(2);
  });

  it('treats a socket constructor that throws as a failed attempt', () => {
    vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    let calls = 0;
    const client = new LiveClient({
      url: 'ws://blocked',
      socketFactory: () => {
        calls += 1;
        throw new Error('SecurityError');
      },
      backoff: { initialMs: 100, maxMs: 100, jitter: 0 },
    });
    client.start();
    expect(client.status.state).toBe('reconnecting');
    vi.advanceTimersByTime(100);
    expect(calls).toBe(2);
    client.stop();
  });
});

describe('backoffDelay', () => {
  const options = { initialMs: 500, maxMs: 15_000, jitter: 0.3 };

  it('doubles per attempt and never exceeds the cap', () => {
    expect([1, 2, 3, 4, 5, 6, 7].map((a) => backoffDelay(a, options, () => 0))).toEqual([
      500, 1_000, 2_000, 4_000, 8_000, 15_000, 15_000,
    ]);
  });

  it('jitter only shortens the delay, by at most the jitter fraction', () => {
    expect(backoffDelay(3, options, () => 0.999_999)).toBe(1_400);
    for (let r = 0; r < 1; r += 0.125) {
      const d = backoffDelay(10, options, () => r);
      expect(d).toBeLessThanOrEqual(15_000);
      expect(d).toBeGreaterThanOrEqual(15_000 * 0.7);
    }
  });
});

describe('liveUrl', () => {
  it('follows the page scheme and host', () => {
    expect(liveUrl({ protocol: 'http:', host: 'localhost:5173' })).toBe('ws://localhost:5173/ws/live');
    expect(liveUrl({ protocol: 'https:', host: 'tank.example' })).toBe('wss://tank.example/ws/live');
  });
});
