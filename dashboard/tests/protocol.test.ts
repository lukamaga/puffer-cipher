import { describe, expect, it } from 'vitest';

import { tamperOneByte } from '../src/lib/aead.ts';
import { formatAgo, formatBits, formatDuration, formatRate } from '../src/lib/format.ts';
import { parseBase64, parseHex, toBase64Url, toHex } from '../src/lib/bytes.ts';
import {
  type FrameData,
  type MessageType,
  parsePulseSummary,
  parseServerMessage,
  restDecoders,
} from '../src/lib/protocol.ts';

const led = {
  state: 'verified',
  match_ratio: 0.97,
  lag_s: 0.14,
  confidence_bits: 51.2,
  expected: '0110',
  observed: '0110',
  window_start: 10.0,
  window_end: 22.8,
};

const fishCard = {
  fish_id: 'f3',
  name: 'Nemo',
  color: '#ff8c42',
  present: true,
  track_id: 7,
  stats: {
    speed_avg: 41.2,
    speed_max: 180,
    activity: 0.73,
    chaos_score: 0.81,
    bits_total: 120_400,
    bits_today: 8_812,
    distance_px: 38_200,
    seen_s: 11_520,
  },
};

const receipt = {
  receipt_version: 1,
  purpose: 'key',
  bytes: 32,
  timestamp_ms: 1793454000000,
  frames: [1200, 1290],
  mode: 'day',
  liveness: { state: 'verified', confidence_bits: 54.2 },
  channels: {
    sensor: { credited_bits: 812.0, share: 0.61, health: 'ok' },
    fish: { credited_bits: 512.5, share: 0.39, health: 'ok' },
    system: { credited_bits: 256.0, share: null, health: 'ok' },
  },
  fish: [{ fish_id: 'f3', name: 'Nemo', bits: 120.5, share: 0.31 }],
};

const channel = { bits_per_s: 81_000, h_per_sample: 5.1, credited_total: 1e7, health: 'ok' };

/** One valid example per message type (the mapped type makes the list exhaustive). */
const examples: { [K in MessageType]: unknown } = {
  hello: {
    version: '0.1.0',
    source: 'synthetic',
    width: 640,
    height: 360,
    fps: 10,
    beacon_public_key: 'ab'.repeat(32),
  },
  frame: {
    index: 1200,
    t: 120.5,
    width: 640,
    height: 360,
    jpeg: '/9j/4AAQ',
    tracks: [
      {
        track_id: 7,
        fish_id: 'f3',
        name: 'Nemo',
        color: '#ff8c42',
        bbox: [10, 20, 60, 50],
        trail: [
          [30, 35],
          [32, 36],
        ],
        speed: 41.2,
        heading: 0.3,
      },
    ],
  },
  entropy: {
    t: 120.5,
    mode: 'day',
    channels: {
      sensor: channel,
      fish: { ...channel, bits_per_s: 92 },
      system: { ...channel, h_per_sample: null },
    },
    pool: { sensor: 4096, fish: 812 },
  },
  bits: { t: 120.5, bits: '01'.repeat(128) },
  health: {
    overall: 'degraded',
    channels: {
      fish: {
        state: 'degraded',
        reasons: ['fish_presence: 5 of 6 fish visible'],
        checks: { rct: 'ok', fish_presence: 'degraded' },
      },
    },
  },
  led,
  fish: { fish: [fishCard] },
  beacon: {
    version: 1,
    index: 3,
    timestamp_ms: 1793454000000,
    period_ms: 60000,
    local_random: '00',
    precommitment: '00',
    prev_output: '00',
    summary: '{}',
    summary_hash: '00',
    signature: '00',
    output: '00',
    public_key: '00',
  },
};

const send = (type: string, data: unknown) => JSON.stringify({ type, data });

describe('parseServerMessage', () => {
  it.each(Object.keys(examples) as MessageType[])('decodes a valid %s message', (type) => {
    const result = parseServerMessage(send(type, examples[type]));
    expect(result).toMatchObject({ ok: true, message: { type } });
  });

  it('keeps the payload intact and typed', () => {
    const result = parseServerMessage(send('frame', examples.frame));
    if (!result.ok || result.message.type !== 'frame') throw new Error('expected a frame');
    const frame: FrameData = result.message.data;
    expect(frame.tracks[0]?.bbox).toEqual([10, 20, 60, 50]);
    expect(frame.tracks[0]?.trail[1]).toEqual([32, 36]);
  });

  it('drops unknown fields so the backend can grow the protocol', () => {
    const result = parseServerMessage(send('bits', { t: 1, bits: '01', future_field: 42 }));
    expect(result).toEqual({ ok: true, message: { type: 'bits', data: { t: 1, bits: '01' } } });
  });

  it('normalises nullable fields: unidentified fish, absent JPEG, numeric version', () => {
    const frame = parseServerMessage(
      send('frame', {
        ...(examples.frame as object),
        jpeg: undefined,
        tracks: [
          {
            track_id: 1,
            fish_id: null,
            name: null,
            color: null,
            bbox: [0, 0, 1, 1],
            trail: [],
            speed: 0,
            heading: 0,
          },
        ],
      }),
    );
    expect(frame.ok && frame.message.type === 'frame' && frame.message.data.jpeg).toBeNull();
    const hello = parseServerMessage(
      send('hello', { ...(examples.hello as object), version: 1, beacon_public_key: null }),
    );
    expect(hello.ok && hello.message.data).toMatchObject({ version: '1', beacon_public_key: null });
  });

  it('classifies malformed envelopes', () => {
    expect(parseServerMessage('{not json')).toMatchObject({ ok: false, reason: 'invalid_json' });
    expect(parseServerMessage('[1,2]')).toMatchObject({ ok: false, reason: 'invalid_envelope' });
    expect(parseServerMessage('{"data":{}}')).toMatchObject({ ok: false, reason: 'invalid_envelope' });
    expect(parseServerMessage(new ArrayBuffer(4))).toMatchObject({ ok: false, reason: 'invalid_envelope' });
    expect(parseServerMessage(send('telemetry', {}))).toEqual({
      ok: false,
      reason: 'unknown_type',
      detail: 'telemetry',
    });
  });

  it.each<[string, MessageType, unknown, string]>([
    ['a string where a number belongs', 'bits', { t: '1', bits: '01' }, 'data.t'],
    ['a non 0/1 bit string', 'bits', { t: 1, bits: '012' }, 'data.bits'],
    ['an unknown health state', 'health', { overall: 'meh', channels: {} }, 'data.overall'],
    [
      'a short bbox',
      'frame',
      {
        ...(examples.frame as object),
        tracks: [
          {
            track_id: 1,
            fish_id: null,
            name: null,
            color: null,
            bbox: [0, 0, 1],
            trail: [],
            speed: 0,
            heading: 0,
          },
        ],
      },
      'data.tracks[0].bbox',
    ],
    [
      'a missing channel',
      'entropy',
      { ...(examples.entropy as object), channels: { sensor: channel, fish: channel } },
      'data.channels.system',
    ],
    ['a fractional frame index', 'frame', { ...(examples.frame as object), index: 1.5 }, 'data.index'],
  ])('rejects %s and names the field', (_name, type, data, path) => {
    const result = parseServerMessage(send(type, data));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toBe('invalid_data');
      expect(result.detail.startsWith(`${path}:`)).toBe(true);
    }
  });

  it('rejects non-finite numbers smuggled in as JSON-like values', () => {
    const result = parseServerMessage('{"type":"bits","data":{"t":1e999,"bits":"0"}}');
    expect(result).toMatchObject({ ok: false, reason: 'invalid_data' });
  });
});

describe('REST decoders', () => {
  it('decodes /api/status', () => {
    const status = restDecoders.status({
      source: 'synthetic',
      mode: 'night',
      fps: 10,
      uptime_s: 61.5,
      frames: 615,
      pool: { sensor: 4096, fish: 120 },
      liveness: led,
      health: { sensor: { channel: 'sensor', state: 'ok', reasons: [], checks: {} } },
    });
    expect(status.health.sensor?.channel).toBe('sensor');
  });

  it('decodes the encrypt demo response with its receipt', () => {
    const r = restDecoders.encrypt({ ciphertext: 'AA', key: 'BB', key_id: 'a1b2c3d4e5f60718', receipt });
    expect(r.receipt.channels.system.share).toBeNull();
    expect(r.receipt.fish[0]?.name).toBe('Nemo');
  });

  it('decodes fish, leaderboard and the beacon key', () => {
    expect(restDecoders.fish([fishCard])[0]?.stats.chaos_score).toBe(0.81);
    expect(
      restDecoders.leaderboard([{ rank: 1, fish_id: 'f3', name: 'Nemo', chaos_score: 0.9, bits: 10 }]),
    ).toHaveLength(1);
    expect(() => restDecoders.publicKey({ algorithm: 'RSA', public_key: 'ab' })).toThrow(/algorithm/);
  });
});

describe('pulse summaries', () => {
  it('parses the documented summary fields and tolerates garbage', () => {
    const summary = parsePulseSummary(
      '{"fish_bits":812,"fish_count":2,"frames":[1,9],"health":{"fish":"ok"},"liveness":"verified",' +
        '"liveness_confidence_bits":54,"mode":"day","sensor_bits":4096,"system_bits":256,' +
        '"top_fish":[{"bits":500,"fish_id":"f3","name":"Nemo"}]}',
    );
    expect(summary?.top_fish[0]?.name).toBe('Nemo');
    expect(parsePulseSummary('{"fish_count": "two"}')).toBeNull();
    expect(parsePulseSummary('nope')).toBeNull();
  });
});

describe('byte encodings', () => {
  it('round-trips hex and rejects malformed input', () => {
    const bytes = new Uint8Array([0, 1, 0xab, 0xff]);
    expect(toHex(bytes)).toBe('0001abff');
    expect(parseHex('0001ABff')).toEqual(bytes);
    expect(parseHex('abc')).toBeNull();
    expect(parseHex('zz')).toBeNull();
    expect(parseHex('00', 2)).toBeNull();
  });

  it('round-trips base64url without padding and accepts standard base64', () => {
    const bytes = new Uint8Array([0xfb, 0xff, 0xfe, 0x00, 0x10]);
    const encoded = toBase64Url(bytes);
    expect(encoded).toBe('-__-ABA');
    expect(parseBase64(encoded)).toEqual(bytes);
    expect(parseBase64('+//+ABA=')).toEqual(bytes);
    expect(parseBase64('a')).toBeNull();
    expect(parseBase64('a$b')).toBeNull();
  });
});

describe('tamperOneByte', () => {
  it('flips exactly one bit, inside the ciphertext body, never the nonce', () => {
    const wire = new Uint8Array(24 + 16 + 10).map((_, i) => i);
    const tampered = tamperOneByte(toBase64Url(wire));
    expect(tampered).not.toBeNull();
    const bytes = parseBase64(tampered?.ciphertext ?? '') as Uint8Array;
    const changed = [...bytes].flatMap((b, i) => (b === wire[i] ? [] : [i]));
    expect(changed).toEqual([tampered?.offset]);
    expect(tampered?.offset).toBeGreaterThanOrEqual(24);
    expect((bytes[changed[0] ?? 0] ?? 0) ^ (wire[changed[0] ?? 0] ?? 0)).toBe(1);
  });

  it('refuses input too short to hold a nonce and a tag', () => {
    expect(tamperOneByte(toBase64Url(new Uint8Array(39)))).toBeNull();
    expect(tamperOneByte('***')).toBeNull();
  });
});

describe('formatting', () => {
  it('labels relative time, bit quantities and durations compactly', () => {
    expect([0, 0.4, 30, 120, 157].map(formatAgo)).toEqual(['now', 'now', '-30s', '-2m', '-2m 37s']);
    expect([950, 81_920, 1_250_000].map((b) => formatBits(b))).toEqual(['950 bit', '81.9 kbit', '1.3 Mbit']);
    expect(formatRate(26)).toBe('26 bit/s');
    expect([42, 245, 11_520, 535_000].map(formatDuration)).toEqual(['42s', '4m 05s', '3h 12m', '6d 04h']);
  });
});

describe('/api/random', () => {
  it('decodes random bytes with their receipt', () => {
    const r = restDecoders.random({ data: 'ab'.repeat(32), receipt });
    expect(r.receipt.purpose).toBe('key');
    expect(() => restDecoders.random({ data: 'ab', receipt: { ...receipt, mode: 'dusk' } })).toThrow(
      /receipt\.mode/,
    );
  });
});

describe('backend edge cases (DESIGN 3.4 / 4.5 / 5)', () => {
  it('accepts [] as "no frames" in receipts and beacon summaries', async () => {
    const { entropyReceipt, parsePulseSummary } = await import('../src/lib/protocol.ts');
    const receipt = entropyReceipt({
      receipt_version: 1,
      purpose: 'key',
      bytes: 32,
      timestamp_ms: 1,
      frames: [],
      mode: 'unknown',
      liveness: { state: 'calibrating', confidence_bits: 0 },
      channels: {
        sensor: { credited_bits: 0, share: null, health: 'startup' },
        fish: { credited_bits: 0, share: null, health: 'startup' },
        system: { credited_bits: 256, share: null, health: 'ok' },
      },
      fish: [],
    });
    expect(receipt.frames).toBeNull();
    expect(receipt.mode).toBe('unknown');
    const summary = parsePulseSummary(
      JSON.stringify({
        fish_count: 0,
        fish_bits: 0,
        sensor_bits: 0,
        system_bits: 256,
        mode: 'day',
        liveness: 'calibrating',
        liveness_confidence_bits: 0,
        health: { sensor: 'startup', fish: 'startup' },
        top_fish: [],
        frames: [],
      }),
    );
    expect(summary?.frames).toBeNull();
  });

  it("accepts '?' for undecodable observed LED bits but not in expected bits", async () => {
    const { parseServerMessage } = await import('../src/lib/protocol.ts');
    const led = (observed: string, expected = '0101') =>
      parseServerMessage(
        JSON.stringify({
          type: 'led',
          data: {
            state: 'verified',
            match_ratio: 0.95,
            lag_s: 0.08,
            confidence_bits: 40,
            expected,
            observed,
            window_start: 0,
            window_end: 12.8,
          },
        }),
      );
    expect(led('01?1').ok).toBe(true);
    expect(led('01x1').ok).toBe(false);
    expect(led('0101', '01?1').ok).toBe(false);
  });
});
