/**
 * Resilient client for the live feed (`/ws/live`, DESIGN.md 7.2).
 *
 * Design choices:
 * - Exponential backoff with jitter, so a restarted backend is not stampeded
 *   by every open dashboard at the same instant.
 * - The backoff only resets after the server has sent a *valid* message,
 *   not merely on `open`: a proxy that accepts the upgrade and drops the
 *   connection immediately would otherwise cause a tight reconnect loop.
 * - A watchdog closes connections that go silent. The server streams frames
 *   at ~10 Hz, so silence means a half-open TCP connection (laptop sleep,
 *   NAT timeout) that the browser may not report for minutes.
 * - Every message is decoded by `parseServerMessage` before any subscriber
 *   sees it; invalid messages are counted, never delivered.
 * - The socket is injected (`SocketFactory`), which keeps the reconnect
 *   state machine testable without a network or a DOM.
 */

import { type MessageMap, type MessageType, parseServerMessage } from './protocol.ts';

export type ConnectionState = 'idle' | 'connecting' | 'open' | 'reconnecting' | 'stopped';

export interface ConnectionStatus {
  state: ConnectionState;
  /** Consecutive attempts that ended without a valid message. */
  attempt: number;
  /** Delay before the next attempt while `reconnecting`, else `null`. */
  retryInMs: number | null;
  /** Messages rejected by the protocol decoder since start. */
  rejected: number;
}

export interface SocketHandlers {
  onOpen(): void;
  onMessage(data: unknown): void;
  onClose(code: number, reason: string): void;
  onError(): void;
}

export interface SocketHandle {
  close(code?: number, reason?: string): void;
}

/** Opens a socket and forwards its events; returns a handle to close it. */
export type SocketFactory = (url: string, handlers: SocketHandlers) => SocketHandle;

export const browserSocketFactory: SocketFactory = (url, handlers) => {
  const socket = new WebSocket(url);
  socket.onopen = () => handlers.onOpen();
  socket.onmessage = (event: MessageEvent<unknown>) => handlers.onMessage(event.data);
  socket.onclose = (event) => handlers.onClose(event.code, event.reason);
  socket.onerror = () => handlers.onError();
  return { close: (code, reason) => socket.close(code, reason) };
};

export interface BackoffOptions {
  initialMs: number;
  maxMs: number;
  /** Fraction of the delay that is randomised downwards (0 = deterministic). */
  jitter: number;
}

export const DEFAULT_BACKOFF: BackoffOptions = { initialMs: 500, maxMs: 15_000, jitter: 0.3 };

/**
 * Delay before reconnect attempt `attempt` (1-based): doubling from
 * `initialMs`, capped at `maxMs`, then reduced by up to `jitter` of itself.
 * Jitter only ever shortens the delay, so `maxMs` is a hard ceiling.
 */
export function backoffDelay(
  attempt: number,
  options: BackoffOptions = DEFAULT_BACKOFF,
  random: () => number = Math.random,
): number {
  const exponent = Math.max(0, attempt - 1);
  const base = Math.min(options.maxMs, options.initialMs * 2 ** exponent);
  return Math.round(base * (1 - options.jitter * random()));
}

export interface LiveClientOptions {
  url: string;
  socketFactory?: SocketFactory;
  backoff?: Partial<BackoffOptions>;
  /** Reconnect when an open connection has been silent for this long. */
  staleAfterMs?: number;
  random?: () => number;
  /** Called once per rejected message (defaults to a console warning). */
  onRejected?: (detail: string) => void;
}

type Handler<T extends MessageType> = (data: MessageMap[T]) => void;
type AnyHandler = (data: MessageMap[MessageType]) => void;

/** Close code for connections we drop ourselves (RFC 6455 "going away"). */
const CLOSE_GOING_AWAY = 1001;

/** Derive the live-feed URL from the page origin (same-origin in dev and prod). */
export function liveUrl(location: Pick<Location, 'protocol' | 'host'>): string {
  const scheme = location.protocol === 'https:' ? 'wss' : 'ws';
  return `${scheme}://${location.host}/ws/live`;
}

export class LiveClient {
  private readonly url: string;
  private readonly factory: SocketFactory;
  private readonly backoff: BackoffOptions;
  private readonly staleAfterMs: number;
  private readonly random: () => number;
  private readonly onRejected: (detail: string) => void;

  private readonly handlers = new Map<MessageType, Set<AnyHandler>>();
  private readonly statusHandlers = new Set<(status: ConnectionStatus) => void>();
  private readonly latestData = new Map<MessageType, MessageMap[MessageType]>();

  private socket: SocketHandle | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private watchdog: ReturnType<typeof setInterval> | null = null;
  private lastMessageAt = 0;
  private currentStatus: ConnectionStatus = {
    state: 'idle',
    attempt: 0,
    retryInMs: null,
    rejected: 0,
  };

  constructor(options: LiveClientOptions) {
    this.url = options.url;
    this.factory = options.socketFactory ?? browserSocketFactory;
    this.backoff = { ...DEFAULT_BACKOFF, ...options.backoff };
    this.staleAfterMs = options.staleAfterMs ?? 10_000;
    this.random = options.random ?? Math.random;
    this.onRejected = options.onRejected ?? ((detail) => console.warn(`[live] rejected message: ${detail}`));
  }

  get status(): ConnectionStatus {
    return this.currentStatus;
  }

  /** Most recent payload of a type, for components that mount late. */
  latest<T extends MessageType>(type: T): MessageMap[T] | undefined {
    return this.latestData.get(type) as MessageMap[T] | undefined;
  }

  /** Subscribe to one message type; returns the unsubscribe function. */
  on<T extends MessageType>(type: T, handler: Handler<T>): () => void {
    let set = this.handlers.get(type);
    if (!set) {
      set = new Set();
      this.handlers.set(type, set);
    }
    const erased = handler as AnyHandler;
    set.add(erased);
    return () => set.delete(erased);
  }

  onStatus(handler: (status: ConnectionStatus) => void): () => void {
    this.statusHandlers.add(handler);
    return () => this.statusHandlers.delete(handler);
  }

  start(): void {
    if (this.currentStatus.state !== 'idle' && this.currentStatus.state !== 'stopped') return;
    this.setStatus({ attempt: 0 });
    this.connect();
  }

  stop(): void {
    this.clearRetry();
    this.stopWatchdog();
    const socket = this.socket;
    this.socket = null;
    socket?.close(CLOSE_GOING_AWAY, 'client stopped');
    this.setStatus({ state: 'stopped', retryInMs: null });
  }

  private connect(): void {
    this.clearRetry();
    this.setStatus({ state: 'connecting', retryInMs: null });
    // Each socket gets handlers bound to *its own* identity, so late events
    // from a socket we already abandoned cannot disturb the current one.
    let handle: SocketHandle | null = null;
    const isCurrent = () => handle !== null && this.socket === handle;
    const handlers: SocketHandlers = {
      onOpen: () => {
        if (!isCurrent()) return;
        this.lastMessageAt = Date.now();
        this.startWatchdog();
        this.setStatus({ state: 'open' });
      },
      onMessage: (data) => {
        if (isCurrent()) this.receive(data);
      },
      onClose: () => {
        if (isCurrent()) this.scheduleReconnect();
      },
      // Browsers always follow `error` with `close`; reconnecting is handled there.
      onError: () => undefined,
    };
    try {
      handle = this.factory(this.url, handlers);
    } catch (e) {
      // e.g. a SecurityError for a blocked URL: treat like a failed attempt.
      console.warn('[live] could not open socket', e);
      this.scheduleReconnect();
      return;
    }
    this.socket = handle;
  }

  private receive(raw: unknown): void {
    this.lastMessageAt = Date.now();
    const parsed = parseServerMessage(raw);
    if (!parsed.ok) {
      if (parsed.reason !== 'unknown_type') {
        this.setStatus({ rejected: this.currentStatus.rejected + 1 });
        this.onRejected(`${parsed.reason}: ${parsed.detail}`);
      }
      return;
    }
    if (this.currentStatus.attempt !== 0) this.setStatus({ attempt: 0 });
    const { type, data } = parsed.message;
    this.latestData.set(type, data);
    for (const handler of this.handlers.get(type) ?? []) {
      try {
        handler(data);
      } catch (e) {
        // One broken panel must not starve the others of data.
        console.error(`[live] ${type} handler failed`, e);
      }
    }
  }

  private scheduleReconnect(): void {
    this.stopWatchdog();
    this.socket = null;
    const attempt = this.currentStatus.attempt + 1;
    const delay = backoffDelay(attempt, this.backoff, this.random);
    this.setStatus({ state: 'reconnecting', attempt, retryInMs: delay });
    this.retryTimer = setTimeout(() => this.connect(), delay);
  }

  private startWatchdog(): void {
    this.stopWatchdog();
    const period = Math.max(250, Math.floor(this.staleAfterMs / 4));
    this.watchdog = setInterval(() => {
      if (Date.now() - this.lastMessageAt < this.staleAfterMs) return;
      const stale = this.socket;
      this.scheduleReconnect();
      stale?.close(CLOSE_GOING_AWAY, 'no data');
    }, period);
  }

  private stopWatchdog(): void {
    if (this.watchdog !== null) clearInterval(this.watchdog);
    this.watchdog = null;
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) clearTimeout(this.retryTimer);
    this.retryTimer = null;
  }

  private setStatus(patch: Partial<ConnectionStatus>): void {
    this.currentStatus = { ...this.currentStatus, ...patch };
    for (const handler of this.statusHandlers) handler(this.currentStatus);
  }
}
