/**
 * Wire protocol of the puffer API (DESIGN.md section 7): TypeScript types for
 * every REST payload (7.1) and WebSocket message (7.2), plus runtime decoders.
 *
 * Why decode at runtime? The socket and REST bodies are untrusted input to a
 * security dashboard. A message that does not match the contract is rejected
 * at the boundary (and counted) instead of leaking `undefined` into a canvas
 * or, worse, into the beacon verifier. Each decoder is typed as
 * `Decoder<Interface>`, so the compiler proves the hand-written interfaces and
 * the runtime checks describe the same shape.
 *
 * Compatibility policy: unknown *fields* are dropped (the backend may add
 * data), unknown message *types* are ignored, but a known field with the wrong
 * type rejects the whole message. Fields the backend can legitimately leave
 * empty (unidentified fish, a disabled beacon) are explicitly nullable.
 */

// ---------------------------------------------------------------------------
// Decoder combinators
// ---------------------------------------------------------------------------

export class DecodeError extends Error {
  readonly path: string;

  constructor(path: string, expected: string, value: unknown) {
    super(`${path || '<root>'}: expected ${expected}, got ${describe(value)}`);
    this.name = 'DecodeError';
    this.path = path;
  }
}

function describe(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  if (typeof value === 'string') return `string ${JSON.stringify(value.slice(0, 24))}`;
  return typeof value;
}

/** Validates `value` and returns it typed, or throws `DecodeError`. */
export type Decoder<T> = (value: unknown, path?: string) => T;

export const string: Decoder<string> = (v, path = '') => {
  if (typeof v !== 'string') throw new DecodeError(path, 'string', v);
  return v;
};

export const number: Decoder<number> = (v, path = '') => {
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new DecodeError(path, 'finite number', v);
  return v;
};

export const integer: Decoder<number> = (v, path = '') => {
  if (typeof v !== 'number' || !Number.isSafeInteger(v)) throw new DecodeError(path, 'integer', v);
  return v;
};

export const boolean: Decoder<boolean> = (v, path = '') => {
  if (typeof v !== 'boolean') throw new DecodeError(path, 'boolean', v);
  return v;
};

export function literal<const T extends readonly (string | number)[]>(...values: T): Decoder<T[number]> {
  return (v, path = '') => {
    if (!values.includes(v as T[number])) {
      throw new DecodeError(path, `one of ${values.join(' | ')}`, v);
    }
    return v as T[number];
  };
}

/** Accepts `null` or a missing field as `null`. */
export function nullable<T>(inner: Decoder<T>): Decoder<T | null> {
  return (v, path = '') => (v === null || v === undefined ? null : inner(v, path));
}

export function array<T>(item: Decoder<T>): Decoder<T[]> {
  return (v, path = '') => {
    if (!Array.isArray(v)) throw new DecodeError(path, 'array', v);
    return v.map((x, i) => item(x, `${path}[${i}]`));
  };
}

export function record<T>(value: Decoder<T>): Decoder<Record<string, T>> {
  return (v, path = '') => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) {
      throw new DecodeError(path, 'object', v);
    }
    const out: Record<string, T> = {};
    for (const [k, x] of Object.entries(v)) out[k] = value(x, `${path}.${k}`);
    return out;
  };
}

export function object<T extends object>(shape: { [K in keyof T]-?: Decoder<T[K]> }): Decoder<T> {
  return (v, path = '') => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) {
      throw new DecodeError(path, 'object', v);
    }
    const source = v as Record<string, unknown>;
    const out: Partial<T> = {};
    for (const key of Object.keys(shape) as (keyof T & string)[]) {
      out[key] = shape[key](source[key], path ? `${path}.${key}` : key);
    }
    return out as T;
  };
}

function tuple<T extends readonly unknown[]>(...items: { [K in keyof T]: Decoder<T[K]> }): Decoder<T> {
  return (v, path = '') => {
    if (!Array.isArray(v) || v.length !== items.length) {
      throw new DecodeError(path, `array of length ${items.length}`, v);
    }
    return items.map((d, i) => d(v[i], `${path}[${i}]`)) as unknown as T;
  };
}

const bitString: Decoder<string> = (v, path = '') => {
  const s = string(v, path);
  if (!/^[01]*$/.test(s)) throw new DecodeError(path, 'string of 0/1', v);
  return s;
};

/** Older backends send the version as a number; normalise to a string. */
const versionString: Decoder<string> = (v, path = '') =>
  typeof v === 'number' ? String(v) : string(v, path);

// ---------------------------------------------------------------------------
// Shared vocabulary (mirrors src/puffer/types.py)
// ---------------------------------------------------------------------------

export const CHANNELS = ['sensor', 'fish', 'system'] as const;
export type ChannelName = (typeof CHANNELS)[number];
/** Channels fed by the camera; `system` (os.urandom) is always present on top. */
export const CAMERA_CHANNELS = ['sensor', 'fish'] as const;
export type CameraChannel = (typeof CAMERA_CHANNELS)[number];

export const HEALTH_STATES = ['ok', 'startup', 'degraded', 'failed'] as const;
export type HealthState = (typeof HEALTH_STATES)[number];
export const healthState = literal(...HEALTH_STATES);

export const LIVENESS_STATES = ['disabled', 'calibrating', 'verified', 'failed'] as const;
export type LivenessState = (typeof LIVENESS_STATES)[number];
export const livenessState = literal(...LIVENESS_STATES);

export type LightMode = 'day' | 'night';
export const lightMode = literal('day', 'night');

export type BBox = readonly [x1: number, y1: number, x2: number, y2: number];
export type Point = readonly [x: number, y: number];
const bbox: Decoder<BBox> = tuple(number, number, number, number);
const point: Decoder<Point> = tuple(number, number);

// ---------------------------------------------------------------------------
// Shared payloads
// ---------------------------------------------------------------------------

/** `types.LivenessStatus` as JSON; also the `led` message. */
export interface LivenessStatus {
  state: LivenessState;
  match_ratio: number;
  lag_s: number;
  confidence_bits: number;
  /** Recent commanded half-symbols, e.g. "0110". */
  expected: string;
  /** Recent observed half-symbols at the best lag. */
  observed: string;
  window_start: number;
  window_end: number;
}

export const livenessStatus: Decoder<LivenessStatus> = object<LivenessStatus>({
  state: livenessState,
  match_ratio: number,
  lag_s: number,
  confidence_bits: number,
  expected: bitString,
  observed: bitString,
  window_start: number,
  window_end: number,
});

export interface ChannelHealth {
  state: HealthState;
  reasons: string[];
  /** check name -> state, e.g. `{"rct": "ok", "fish_presence": "degraded"}` */
  checks: Record<string, HealthState>;
}

const channelHealth: Decoder<ChannelHealth> = object<ChannelHealth>({
  state: healthState,
  reasons: array(string),
  checks: record(healthState),
});

/** `types.HealthReport` as served by `/api/status` (carries its channel name). */
export interface HealthReport extends ChannelHealth {
  channel: string | null;
}

const healthReport: Decoder<HealthReport> = object<HealthReport>({
  channel: nullable(string),
  state: healthState,
  reasons: array(string),
  checks: record(healthState),
});

export interface FishStats {
  speed_avg: number;
  speed_max: number;
  activity: number;
  chaos_score: number;
  bits_total: number;
  bits_today: number;
  distance_px: number;
  seen_s: number;
}

export interface FishCard {
  fish_id: string;
  name: string;
  /** "#rrggbb", assigned by the backend and stable per fish. */
  color: string;
  present: boolean;
  track_id: number | null;
  stats: FishStats;
}

export const fishCard: Decoder<FishCard> = object<FishCard>({
  fish_id: string,
  name: string,
  color: string,
  present: boolean,
  track_id: nullable(integer),
  stats: object<FishStats>({
    speed_avg: number,
    speed_max: number,
    activity: number,
    chaos_score: number,
    bits_total: number,
    bits_today: number,
    distance_px: number,
    seen_s: number,
  }),
});

/** `EntropyReceipt.to_dict()` (DESIGN 4.5). Never secret. */
export interface EntropyReceipt {
  receipt_version: number;
  purpose: string;
  bytes: number;
  timestamp_ms: number;
  frames: readonly [first: number, last: number] | null;
  mode: LightMode;
  liveness: { state: LivenessState; confidence_bits: number };
  channels: Record<ChannelName, ReceiptChannel>;
  fish: ReceiptFish[];
}

export interface ReceiptChannel {
  credited_bits: number;
  /** Share over camera channels only; `null` for `system`. */
  share: number | null;
  health: HealthState;
}

export interface ReceiptFish {
  fish_id: string;
  name: string | null;
  bits: number;
  share: number;
}

const receiptChannel: Decoder<ReceiptChannel> = object<ReceiptChannel>({
  credited_bits: number,
  share: nullable(number),
  health: healthState,
});

export const entropyReceipt: Decoder<EntropyReceipt> = object<EntropyReceipt>({
  receipt_version: integer,
  purpose: string,
  bytes: integer,
  timestamp_ms: integer,
  frames: nullable(tuple(integer, integer)),
  mode: lightMode,
  liveness: object({ state: livenessState, confidence_bits: number }),
  channels: object<Record<ChannelName, ReceiptChannel>>({
    sensor: receiptChannel,
    fish: receiptChannel,
    system: receiptChannel,
  }),
  fish: array(object<ReceiptFish>({ fish_id: string, name: nullable(string), bits: number, share: number })),
});

/**
 * Beacon pulse JSON (DESIGN 3.4). The decoder checks JSON types only; byte
 * lengths and hex syntax are the verifier's job, so a malformed pulse is
 * still displayed - with the exact check it fails - instead of vanishing.
 */
export interface BeaconPulse {
  version: number;
  index: number;
  timestamp_ms: number;
  period_ms: number;
  local_random: string;
  precommitment: string;
  prev_output: string;
  summary: string;
  summary_hash: string;
  signature: string;
  output: string;
  public_key: string;
}

export const beaconPulse: Decoder<BeaconPulse> = object<BeaconPulse>({
  version: integer,
  index: integer,
  timestamp_ms: integer,
  period_ms: integer,
  local_random: string,
  precommitment: string,
  prev_output: string,
  summary: string,
  summary_hash: string,
  signature: string,
  output: string,
  public_key: string,
});

/** The parsed `summary` string of a pulse (DESIGN 3.4). Integers only, by design. */
export interface PulseSummary {
  fish_count: number;
  fish_bits: number;
  sensor_bits: number;
  system_bits: number;
  mode: LightMode;
  liveness: LivenessState;
  liveness_confidence_bits: number;
  health: Record<string, HealthState>;
  top_fish: { fish_id: string; name: string; bits: number }[];
  frames: readonly [first: number, last: number] | null;
}

export const pulseSummary: Decoder<PulseSummary> = object<PulseSummary>({
  fish_count: integer,
  fish_bits: integer,
  sensor_bits: integer,
  system_bits: integer,
  mode: lightMode,
  liveness: livenessState,
  liveness_confidence_bits: integer,
  health: record(healthState),
  top_fish: array(object({ fish_id: string, name: string, bits: integer })),
  frames: nullable(tuple(integer, integer)),
});

/** Parse a pulse's summary for display. The hash check never depends on this. */
export function parsePulseSummary(summary: string): PulseSummary | null {
  try {
    return pulseSummary(JSON.parse(summary));
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// WebSocket /ws/live (DESIGN 7.2)
// ---------------------------------------------------------------------------

export interface HelloData {
  version: string;
  source: string;
  width: number;
  height: number;
  fps: number;
  /** `null` when the beacon is disabled. */
  beacon_public_key: string | null;
}

export interface FrameTrack {
  track_id: number;
  /** `null` until re-identification has matched the track to a known fish. */
  fish_id: string | null;
  name: string | null;
  color: string | null;
  bbox: BBox;
  /** Recent centres in frame pixels, oldest first. */
  trail: Point[];
  speed: number;
  heading: number;
}

export interface FrameData {
  index: number;
  t: number;
  width: number;
  height: number;
  /** Base64 JPEG; `null`/empty when the source sends tracks only (mock, headless). */
  jpeg: string | null;
  tracks: FrameTrack[];
}

export interface ChannelEntropy {
  /** Credited bits per second (EMA). */
  bits_per_s: number;
  /** Rolling min-entropy estimate per sample; `null` when not estimated (system). */
  h_per_sample: number | null;
  credited_total: number;
  health: HealthState;
}

export interface EntropyData {
  t: number;
  mode: LightMode;
  channels: Record<ChannelName, ChannelEntropy>;
  /** Current (undrawn) credit per camera channel, in bits. */
  pool: Record<CameraChannel, number>;
}

export interface BitsData {
  t: number;
  /** 256 display bits from the public `display` stream. Never key material. */
  bits: string;
}

export interface HealthData {
  overall: HealthState;
  channels: Record<string, ChannelHealth>;
}

export interface FishData {
  fish: FishCard[];
}

export interface MessageMap {
  hello: HelloData;
  frame: FrameData;
  entropy: EntropyData;
  bits: BitsData;
  health: HealthData;
  led: LivenessStatus;
  fish: FishData;
  beacon: BeaconPulse;
}

export type MessageType = keyof MessageMap;
export type ServerMessage = { [K in MessageType]: { type: K; data: MessageMap[K] } }[MessageType];

const channelEntropy: Decoder<ChannelEntropy> = object<ChannelEntropy>({
  bits_per_s: number,
  h_per_sample: nullable(number),
  credited_total: number,
  health: healthState,
});

export const messageDecoders: { [K in MessageType]: Decoder<MessageMap[K]> } = {
  hello: object<HelloData>({
    version: versionString,
    source: string,
    width: integer,
    height: integer,
    fps: number,
    beacon_public_key: nullable(string),
  }),
  frame: object<FrameData>({
    index: integer,
    t: number,
    width: integer,
    height: integer,
    jpeg: nullable(string),
    tracks: array(
      object<FrameTrack>({
        track_id: integer,
        fish_id: nullable(string),
        name: nullable(string),
        color: nullable(string),
        bbox,
        trail: array(point),
        speed: number,
        heading: number,
      }),
    ),
  }),
  entropy: object<EntropyData>({
    t: number,
    mode: lightMode,
    channels: object<Record<ChannelName, ChannelEntropy>>({
      sensor: channelEntropy,
      fish: channelEntropy,
      system: channelEntropy,
    }),
    pool: object<Record<CameraChannel, number>>({ sensor: number, fish: number }),
  }),
  bits: object<BitsData>({ t: number, bits: bitString }),
  health: object<HealthData>({ overall: healthState, channels: record(channelHealth) }),
  led: livenessStatus,
  fish: object<FishData>({ fish: array(fishCard) }),
  beacon: beaconPulse,
};

export type ParseResult =
  | { ok: true; message: ServerMessage }
  | {
      ok: false;
      reason: 'invalid_json' | 'invalid_envelope' | 'unknown_type' | 'invalid_data';
      detail: string;
    };

function isMessageType(type: string): type is MessageType {
  return Object.hasOwn(messageDecoders, type);
}

/** Parse one WebSocket text frame. Never throws. */
export function parseServerMessage(raw: unknown): ParseResult {
  if (typeof raw !== 'string') {
    return { ok: false, reason: 'invalid_envelope', detail: 'binary frames are not part of the protocol' };
  }
  let envelope: unknown;
  try {
    envelope = JSON.parse(raw);
  } catch (e) {
    return { ok: false, reason: 'invalid_json', detail: String(e) };
  }
  if (typeof envelope !== 'object' || envelope === null || Array.isArray(envelope)) {
    return { ok: false, reason: 'invalid_envelope', detail: 'message is not a JSON object' };
  }
  const { type, data } = envelope as { type?: unknown; data?: unknown };
  if (typeof type !== 'string') {
    return { ok: false, reason: 'invalid_envelope', detail: 'missing string "type"' };
  }
  if (!isMessageType(type)) {
    return { ok: false, reason: 'unknown_type', detail: type };
  }
  try {
    const decoded = messageDecoders[type](data, 'data');
    return { ok: true, message: { type, data: decoded } as ServerMessage };
  } catch (e) {
    return { ok: false, reason: 'invalid_data', detail: e instanceof Error ? e.message : String(e) };
  }
}

// ---------------------------------------------------------------------------
// REST /api (DESIGN 7.1)
// ---------------------------------------------------------------------------

export interface StatusResponse {
  source: string;
  mode: LightMode;
  fps: number;
  uptime_s: number;
  frames: number;
  pool: Record<string, number>;
  liveness: LivenessStatus;
  health: Record<string, HealthReport>;
}

export interface RandomResponse {
  data: string;
  receipt: EntropyReceipt;
}

export interface EncryptRequest {
  plaintext: string;
}

/** DEMO ONLY: the key travels to the browser. */
export interface EncryptResponse {
  /** base64url(nonce || ciphertext), XChaCha20-Poly1305, aad "puffer-msg-v1" (DESIGN 3.3). */
  ciphertext: string;
  /** base64url, 32 bytes. */
  key: string;
  key_id: string;
  receipt: EntropyReceipt;
}

export interface DecryptRequest {
  ciphertext: string;
  key: string;
}

export interface DecryptResponse {
  plaintext: string;
}

export interface LeaderboardEntry {
  rank: number;
  fish_id: string;
  name: string;
  chaos_score: number;
  bits: number;
}

export interface PublicKeyResponse {
  algorithm: 'Ed25519';
  public_key: string;
}

export const restDecoders = {
  status: object<StatusResponse>({
    source: string,
    mode: lightMode,
    fps: number,
    uptime_s: number,
    frames: integer,
    pool: record(number),
    liveness: livenessStatus,
    health: record(healthReport),
  }),
  random: object<RandomResponse>({ data: string, receipt: entropyReceipt }),
  encrypt: object<EncryptResponse>({
    ciphertext: string,
    key: string,
    key_id: string,
    receipt: entropyReceipt,
  }),
  decrypt: object<DecryptResponse>({ plaintext: string }),
  fish: array(fishCard),
  leaderboard: array(
    object<LeaderboardEntry>({
      rank: integer,
      fish_id: string,
      name: string,
      chaos_score: number,
      bits: number,
    }),
  ),
  publicKey: object<PublicKeyResponse>({ algorithm: literal('Ed25519'), public_key: string }),
  pulse: beaconPulse,
  pulses: array(beaconPulse),
} satisfies Record<string, Decoder<unknown>>;
