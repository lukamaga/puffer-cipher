/**
 * Signing side of the puffer beacon, for the mock server and tests only.
 *
 * The real chain is produced by `puffer.beacon` (Python, libsodium). This
 * builder exists so the dashboard can be developed and screenshotted without
 * the backend while the in-browser verifier still checks a genuine,
 * correctly signed chain - mock data must not be "verified" by special
 * casing. It reuses the browser's message layout (`src/lib/beacon.ts`) so
 * there is exactly one TypeScript definition of the 260-byte message.
 *
 * Node imports this file directly (native TypeScript type stripping), which
 * is why imports carry explicit `.ts` extensions.
 */

import { ed25519 } from '@noble/curves/ed25519.js';
import { blake2b } from '@noble/hashes/blake2.js';

import {
  encodePulseMessage,
  OUTPUT_BYTES,
  precommitmentOf,
  pulseOutput,
  RANDOM_BYTES,
  summaryHash,
} from '../src/lib/beacon.ts';
import { toHex, utf8 } from '../src/lib/bytes.ts';
import type { BeaconPulse } from '../src/lib/protocol.ts';

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

/**
 * `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
 *
 * `JSON.stringify` already matches Python for the summary's value types
 * (ints, strings, bools, lists, objects) once keys are sorted; the remaining
 * difference is `ensure_ascii`, which escapes every non-ASCII UTF-16 code
 * unit as a lowercase `\uXXXX`.
 */
export function canonicalJson(value: JsonValue): string {
  const sortKeys = (v: JsonValue): JsonValue => {
    if (Array.isArray(v)) return v.map(sortKeys);
    if (v !== null && typeof v === 'object') {
      return Object.fromEntries(
        Object.keys(v)
          .sort()
          .map((k) => [k, sortKeys(v[k] as JsonValue)]),
      );
    }
    if (typeof v === 'number' && !Number.isSafeInteger(v)) {
      throw new TypeError(`beacon summaries carry integers only, got ${v}`);
    }
    return v;
  };
  return JSON.stringify(sortKeys(value)).replace(
    /[\u0080-￿]/g,
    (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`,
  );
}

export interface ChainOptions {
  /** 32-byte Ed25519 seed (the PyNaCl `SigningKey(seed)` convention). */
  seed: Uint8Array;
  periodMs: number;
  /** Source of `local_random`; must return fresh 64-byte values. */
  randomBytes: (n: number) => Uint8Array;
}

/**
 * Appends pulses to a hash chain. `local_random` for pulse i+1 is drawn when
 * pulse i is signed, so pulse i can precommit to it - the same order the
 * real beacon uses.
 */
export class BeaconChain {
  readonly publicKeyHex: string;
  readonly pulses: BeaconPulse[] = [];
  private readonly seed: Uint8Array;
  private readonly periodMs: number;
  private readonly randomBytes: (n: number) => Uint8Array;
  private nextRandom: Uint8Array;
  private prevOutput: Uint8Array = new Uint8Array(OUTPUT_BYTES);

  constructor(options: ChainOptions) {
    this.seed = options.seed;
    this.periodMs = options.periodMs;
    this.randomBytes = options.randomBytes;
    this.publicKeyHex = toHex(ed25519.getPublicKey(this.seed));
    this.nextRandom = this.randomBytes(RANDOM_BYTES);
  }

  get latest(): BeaconPulse | undefined {
    return this.pulses.at(-1);
  }

  append(summary: JsonValue, timestampMs: number): BeaconPulse {
    const summaryText = canonicalJson(summary);
    const localRandom = this.nextRandom;
    this.nextRandom = this.randomBytes(RANDOM_BYTES);
    const fields = {
      index: this.pulses.length,
      timestampMs,
      periodMs: this.periodMs,
      localRandom,
      precommitment: precommitmentOf(this.nextRandom),
      prevOutput: this.prevOutput,
      summaryHash: summaryHash(summaryText),
    };
    const message = encodePulseMessage(fields);
    const signature = ed25519.sign(message, this.seed);
    const output = pulseOutput(message, signature);
    const pulse: BeaconPulse = {
      version: 1,
      index: fields.index,
      timestamp_ms: timestampMs,
      period_ms: this.periodMs,
      local_random: toHex(localRandom),
      precommitment: toHex(fields.precommitment),
      prev_output: toHex(fields.prevOutput),
      summary: summaryText,
      summary_hash: toHex(fields.summaryHash),
      signature: toHex(signature),
      output: toHex(output),
      public_key: this.publicKeyHex,
    };
    this.prevOutput = output;
    this.pulses.push(pulse);
    return pulse;
  }
}

/** Deterministic 32-byte seed from a label, so the mock's public key is stable across runs. */
export function seedFromLabel(label: string): Uint8Array {
  return blake2b(utf8(label), { dkLen: 32 });
}
