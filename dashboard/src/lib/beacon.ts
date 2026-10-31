/**
 * Puffer randomness beacon: pulse message construction and verification,
 * byte-for-byte compatible with `puffer.beacon` (DESIGN.md section 3.4).
 *
 * The point of a public beacon is that nobody has to trust the server that
 * serves it. So the dashboard does not display the backend's opinion of a
 * pulse; it re-derives every claim in the browser from the raw JSON:
 *
 *   M = "PUFFER-BEACON-V1" || u64 index || u64 timestamp_ms || u32 period_ms
 *       || local_random(64) || precommitment(64) || prev_output(64)
 *       || summary_hash(32)                                  -> 260 bytes
 *   signature = Ed25519(M), output = BLAKE2b-512(M || signature)
 *
 * Primitives come only from audited libraries (`@noble/curves` Ed25519,
 * `@noble/hashes` BLAKE2b); this module only defines the byte layout and the
 * order in which checks are reported, which must match the Python verifier
 * so the shared test vectors produce the same error codes.
 *
 * Signatures are checked in strict RFC 8032 mode (`zip215: false`) - the
 * closest match to libsodium's `crypto_sign_verify_detached`, which rejects
 * non-canonical encodings that ZIP-215 would accept. A pulse valid here is
 * therefore valid for the Python side as well.
 */

import { ed25519 } from '@noble/curves/ed25519.js';
import { blake2b } from '@noble/hashes/blake2.js';

import { bytesEqual, concatBytes, isAllZero, parseHex, utf8 } from './bytes.ts';
import type { BeaconPulse } from './protocol.ts';

export const BEACON_DOMAIN = 'PUFFER-BEACON-V1';
export const MESSAGE_BYTES = 260;
export const RANDOM_BYTES = 64;
export const SUMMARY_HASH_BYTES = 32;
export const SIGNATURE_BYTES = 64;
export const OUTPUT_BYTES = 64;
export const PUBLIC_KEY_BYTES = 32;

const DOMAIN_BYTES = utf8(BEACON_DOMAIN);
const U32_MAX = 0xffff_ffff;

/** Byte offsets inside M; exported so tests can assert the layout independently. */
export const MESSAGE_LAYOUT = {
  domain: [0, 16],
  index: [16, 24],
  timestampMs: [24, 32],
  periodMs: [32, 36],
  localRandom: [36, 100],
  precommitment: [100, 164],
  prevOutput: [164, 228],
  summaryHash: [228, 260],
} as const satisfies Record<string, readonly [number, number]>;

export interface PulseMessageFields {
  index: number;
  timestampMs: number;
  periodMs: number;
  localRandom: Uint8Array;
  precommitment: Uint8Array;
  prevOutput: Uint8Array;
  summaryHash: Uint8Array;
}

function checkLength(name: string, bytes: Uint8Array, expected: number): void {
  if (bytes.length !== expected) {
    throw new RangeError(`${name} must be ${expected} bytes, got ${bytes.length}`);
  }
}

function checkUint(name: string, value: number, max: number): void {
  if (!Number.isSafeInteger(value) || value < 0 || value > max) {
    throw new RangeError(`${name} must be an integer in [0, ${max}], got ${value}`);
  }
}

/**
 * Build the 260-byte signed message M. Integers are big-endian; u64 fields go
 * through BigInt because JS numbers are only exact up to 2^53 (JSON pulses
 * never exceed that, so `index`/`timestamp_ms` arrive as plain numbers).
 * Throws `RangeError` on malformed fields - callers holding untrusted data
 * should go through `verifyPulse`, which never throws.
 */
export function encodePulseMessage(fields: PulseMessageFields): Uint8Array {
  checkUint('index', fields.index, Number.MAX_SAFE_INTEGER);
  checkUint('timestampMs', fields.timestampMs, Number.MAX_SAFE_INTEGER);
  checkUint('periodMs', fields.periodMs, U32_MAX);
  checkLength('localRandom', fields.localRandom, RANDOM_BYTES);
  checkLength('precommitment', fields.precommitment, RANDOM_BYTES);
  checkLength('prevOutput', fields.prevOutput, OUTPUT_BYTES);
  checkLength('summaryHash', fields.summaryHash, SUMMARY_HASH_BYTES);

  const m = new Uint8Array(MESSAGE_BYTES);
  const view = new DataView(m.buffer);
  const L = MESSAGE_LAYOUT;
  m.set(DOMAIN_BYTES, L.domain[0]);
  view.setBigUint64(L.index[0], BigInt(fields.index), false);
  view.setBigUint64(L.timestampMs[0], BigInt(fields.timestampMs), false);
  view.setUint32(L.periodMs[0], fields.periodMs, false);
  m.set(fields.localRandom, L.localRandom[0]);
  m.set(fields.precommitment, L.precommitment[0]);
  m.set(fields.prevOutput, L.prevOutput[0]);
  m.set(fields.summaryHash, L.summaryHash[0]);
  return m;
}

/** BLAKE2b-256 of the exact UTF-8 bytes of the summary string. */
export function summaryHash(summary: string): Uint8Array {
  return blake2b(utf8(summary), { dkLen: SUMMARY_HASH_BYTES });
}

/** output = BLAKE2b-512(M || signature). */
export function pulseOutput(message: Uint8Array, signature: Uint8Array): Uint8Array {
  return blake2b(concatBytes(message, signature), { dkLen: OUTPUT_BYTES });
}

/** precommitment published in pulse i = BLAKE2b-512(local_random of pulse i+1). */
export function precommitmentOf(localRandom: Uint8Array): Uint8Array {
  return blake2b(localRandom, { dkLen: RANDOM_BYTES });
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/**
 * Error codes shared with the Python verifier (`puffer.beacon.pulse.ErrorCode`),
 * in the order both implementations report them. `error` is the first
 * failing code in this order; `errors` lists all of them so the UI can show
 * one badge per check.
 */
export const BEACON_ERRORS = [
  'wrong_public_key',
  'bad_summary_hash',
  'bad_signature',
  'bad_output',
  'bad_genesis',
  'bad_index',
  'bad_timestamp',
  'broken_link',
  'bad_precommitment',
] as const;
export type BeaconError = (typeof BEACON_ERRORS)[number];

export const PULSE_CHECKS = [
  'public_key',
  'summary_hash',
  'signature',
  'output',
  'genesis',
  'index',
  'timestamp',
  'link',
  'precommitment',
] as const;
export type PulseCheck = (typeof PULSE_CHECKS)[number];

/** `skip`: the check does not apply (no trusted key, no previous pulse, not genesis). */
export type CheckOutcome = 'pass' | 'fail' | 'skip';

const ERROR_OF: Record<PulseCheck, BeaconError> = {
  public_key: 'wrong_public_key',
  summary_hash: 'bad_summary_hash',
  signature: 'bad_signature',
  output: 'bad_output',
  index: 'bad_index',
  timestamp: 'bad_timestamp',
  link: 'broken_link',
  precommitment: 'bad_precommitment',
  genesis: 'bad_genesis',
};

export interface PulseVerification {
  valid: boolean;
  error: BeaconError | null;
  errors: BeaconError[];
  checks: Record<PulseCheck, CheckOutcome>;
}

export interface VerifyOptions {
  /**
   * Trusted beacon key (hex or bytes). The signature is only ever checked
   * against a *trusted* key, never blindly against the key a pulse claims:
   * this key when given, otherwise the previous pulse's key (so a chain
   * cannot silently switch keys), otherwise - for a lone pulse - its own
   * key, which proves internal consistency only (`public_key` is skipped).
   */
  publicKey?: string | Uint8Array | null;
  /** Pulse `index - 1`, enabling the chain checks. */
  prev?: BeaconPulse | null;
}

const outcome = (ok: boolean): CheckOutcome => (ok ? 'pass' : 'fail');

function trustedKeyBytes(key: string | Uint8Array): Uint8Array | null {
  if (typeof key !== 'string') return key.length === PUBLIC_KEY_BYTES ? key : null;
  return parseHex(key, PUBLIC_KEY_BYTES);
}

function verifySignature(signature: Uint8Array, message: Uint8Array, key: Uint8Array): boolean {
  try {
    return ed25519.verify(signature, message, key, { zip215: false });
  } catch {
    // Malformed point encodings throw; for a verifier that is simply "invalid".
    return false;
  }
}

/** Rebuild M from a pulse's JSON fields, or `null` if any field is malformed. */
export function pulseMessage(pulse: BeaconPulse): Uint8Array | null {
  const localRandom = parseHex(pulse.local_random, RANDOM_BYTES);
  const precommitment = parseHex(pulse.precommitment, RANDOM_BYTES);
  const prevOutput = parseHex(pulse.prev_output, OUTPUT_BYTES);
  const summary = parseHex(pulse.summary_hash, SUMMARY_HASH_BYTES);
  if (!localRandom || !precommitment || !prevOutput || !summary) return null;
  try {
    return encodePulseMessage({
      index: pulse.index,
      timestampMs: pulse.timestamp_ms,
      periodMs: pulse.period_ms,
      localRandom,
      precommitment,
      prevOutput,
      summaryHash: summary,
    });
  } catch {
    return null;
  }
}

/**
 * Verify one pulse (DESIGN 3.4 steps 1-5). Every check runs, so a tampered
 * pulse shows exactly which claims break; a check whose inputs are malformed
 * fails rather than throwing. Never throws.
 */
export function verifyPulse(pulse: BeaconPulse, options: VerifyOptions = {}): PulseVerification {
  const { prev = null, publicKey = null } = options;
  const checks = Object.fromEntries(PULSE_CHECKS.map((c) => [c, 'skip'])) as Record<PulseCheck, CheckOutcome>;

  const claimedKey = parseHex(pulse.public_key, PUBLIC_KEY_BYTES);
  let verifyingKey = claimedKey;
  const trustedSource = publicKey ?? prev?.public_key ?? null;
  if (trustedSource !== null) {
    const trusted = trustedKeyBytes(trustedSource);
    checks.public_key = outcome(!!trusted && !!claimedKey && bytesEqual(trusted, claimedKey));
    verifyingKey = trusted;
  }

  const claimedSummaryHash = parseHex(pulse.summary_hash, SUMMARY_HASH_BYTES);
  checks.summary_hash = outcome(
    !!claimedSummaryHash && bytesEqual(claimedSummaryHash, summaryHash(pulse.summary)),
  );

  const message = pulseMessage(pulse);
  const signature = parseHex(pulse.signature, SIGNATURE_BYTES);
  checks.signature = outcome(
    !!message && !!signature && !!verifyingKey && verifySignature(signature, message, verifyingKey),
  );

  const output = parseHex(pulse.output, OUTPUT_BYTES);
  checks.output = outcome(
    !!message && !!signature && !!output && bytesEqual(output, pulseOutput(message, signature)),
  );

  if (pulse.index === 0) {
    const prevOutput = parseHex(pulse.prev_output, OUTPUT_BYTES);
    checks.genesis = outcome(!!prevOutput && isAllZero(prevOutput));
  }

  if (prev) {
    checks.index = outcome(pulse.index === prev.index + 1);
    checks.timestamp = outcome(pulse.timestamp_ms > prev.timestamp_ms);
    const prevOutput = parseHex(pulse.prev_output, OUTPUT_BYTES);
    const prevClaimedOutput = parseHex(prev.output, OUTPUT_BYTES);
    checks.link = outcome(!!prevOutput && !!prevClaimedOutput && bytesEqual(prevOutput, prevClaimedOutput));
    const localRandom = parseHex(pulse.local_random, RANDOM_BYTES);
    const commitment = parseHex(prev.precommitment, RANDOM_BYTES);
    checks.precommitment = outcome(
      !!localRandom && !!commitment && bytesEqual(precommitmentOf(localRandom), commitment),
    );
  }

  const errors = PULSE_CHECKS.filter((c) => checks[c] === 'fail').map((c) => ERROR_OF[c]);
  return { valid: errors.length === 0, error: errors[0] ?? null, errors, checks };
}

export interface ChainVerification {
  valid: boolean;
  results: PulseVerification[];
  /** First failing pulse (array position and pulse index) with its first error. */
  firstFailure: { position: number; index: number; error: BeaconError } | null;
}

/**
 * Verify consecutive pulses (ascending index), each against its predecessor
 * in the list. The first pulse is checked standalone unless `anchor` (the
 * pulse before it) is supplied.
 */
export function verifyChain(
  pulses: readonly BeaconPulse[],
  options: { publicKey?: string | Uint8Array | null; anchor?: BeaconPulse | null } = {},
): ChainVerification {
  const results: PulseVerification[] = [];
  let firstFailure: ChainVerification['firstFailure'] = null;
  let prev = options.anchor ?? null;
  for (const [position, pulse] of pulses.entries()) {
    const result = verifyPulse(pulse, { publicKey: options.publicKey ?? null, prev });
    results.push(result);
    if (!firstFailure && result.error) {
      firstFailure = { position, index: pulse.index, error: result.error };
    }
    prev = pulse;
  }
  return { valid: firstFailure === null, results, firstFailure };
}
