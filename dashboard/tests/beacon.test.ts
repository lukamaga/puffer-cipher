/**
 * Self-contained beacon verifier tests.
 *
 * Pulses are built *in this file*: the 260-byte message is assembled by hand
 * with explicit big-endian arithmetic (not the DataView code under test) and
 * signed with @noble Ed25519. A golden pulse computed with PyNaCl/libsodium
 * and hashlib (`fixtures/golden-pulse.json`) anchors the layout to the Python
 * implementation even before the shared vector file exists.
 */
import { readFileSync } from 'node:fs';

import { ed25519 } from '@noble/curves/ed25519.js';
import { blake2b } from '@noble/hashes/blake2.js';
import { describe, expect, it } from 'vitest';

import { BeaconChain, canonicalJson, seedFromLabel } from '../mock/beacon-chain.ts';
import {
  BEACON_ERRORS,
  encodePulseMessage,
  MESSAGE_BYTES,
  MESSAGE_LAYOUT,
  pulseMessage,
  verifyChain,
  verifyPulse,
} from '../src/lib/beacon.ts';
import { toHex } from '../src/lib/bytes.ts';
import type { BeaconPulse } from '../src/lib/protocol.ts';

// --- independent pulse construction ----------------------------------------

const hex = (bytes: Uint8Array) => Buffer.from(bytes).toString('hex');
const unhex = (s: string) => new Uint8Array(Buffer.from(s, 'hex'));
const b2b = (data: Uint8Array, dkLen: number) => blake2b(data, { dkLen });

function uintBE(value: number, width: number): Uint8Array {
  const out = new Uint8Array(width);
  let v = BigInt(value);
  for (let i = width - 1; i >= 0; i--) {
    out[i] = Number(v & 0xffn);
    v >>= 8n;
  }
  return out;
}

function handMessage(p: Omit<BeaconPulse, 'signature' | 'output' | 'public_key'>): Uint8Array {
  return Buffer.concat([
    Buffer.from('PUFFER-BEACON-V1', 'ascii'),
    uintBE(p.index, 8),
    uintBE(p.timestamp_ms, 8),
    uintBE(p.period_ms, 4),
    unhex(p.local_random),
    unhex(p.precommitment),
    unhex(p.prev_output),
    unhex(p.summary_hash),
  ]);
}

const SECRET = b2b(new TextEncoder().encode('vitest beacon key'), 32);
const PUBLIC = hex(ed25519.getPublicKey(SECRET));
const OTHER_SECRET = b2b(new TextEncoder().encode('an attacker key'), 32);
const OTHER_PUBLIC = hex(ed25519.getPublicKey(OTHER_SECRET));
const randomFor = (i: number) => b2b(new TextEncoder().encode(`local-random-${i}`), 64);

interface PulseSpec {
  index: number;
  timestamp_ms?: number;
  prev_output?: string;
  local_random?: Uint8Array;
  next_random?: Uint8Array;
  summary?: string;
  secret?: Uint8Array;
}

/** Sign a pulse from explicit fields - the way an honest (or forging) beacon would. */
function signPulse(spec: PulseSpec): BeaconPulse {
  const secret = spec.secret ?? SECRET;
  const summary = spec.summary ?? `{"fish_count":${spec.index % 5},"mode":"day"}`;
  const fields = {
    version: 1,
    index: spec.index,
    timestamp_ms: spec.timestamp_ms ?? 1_793_454_000_000 + spec.index * 60_000,
    period_ms: 60_000,
    local_random: hex(spec.local_random ?? randomFor(spec.index)),
    precommitment: hex(b2b(spec.next_random ?? randomFor(spec.index + 1), 64)),
    prev_output: spec.prev_output ?? '00'.repeat(64),
    summary,
    summary_hash: hex(b2b(new TextEncoder().encode(summary), 32)),
  };
  const message = handMessage(fields);
  const signature = ed25519.sign(message, secret);
  return {
    ...fields,
    signature: hex(signature),
    output: hex(b2b(Buffer.concat([message, signature]), 64)),
    public_key: hex(ed25519.getPublicKey(secret)),
  };
}

function honestChain(length: number): BeaconPulse[] {
  const chain: BeaconPulse[] = [];
  for (let i = 0; i < length; i++) {
    chain.push(signPulse({ index: i, prev_output: chain.at(-1)?.output }));
  }
  return chain;
}

const flipHexByte = (s: string, byte = 0) =>
  s.slice(0, 2 * byte) +
  (Number.parseInt(s.slice(2 * byte, 2 * byte + 2), 16) ^ 0x01).toString(16).padStart(2, '0') +
  s.slice(2 * byte + 2);

// --- tests ------------------------------------------------------------------

describe('message layout', () => {
  it('matches the libsodium/hashlib golden pulse byte for byte', () => {
    const golden = JSON.parse(
      readFileSync(new URL('./fixtures/golden-pulse.json', import.meta.url), 'utf8'),
    ) as { message: string; secret_seed: string; pulse: BeaconPulse };
    const message = pulseMessage(golden.pulse);
    expect(message && hex(message)).toBe(golden.message);
    // Ed25519 is deterministic: @noble must reproduce libsodium's signature exactly.
    expect(hex(ed25519.sign(unhex(golden.message), unhex(golden.secret_seed)))).toBe(golden.pulse.signature);
    expect(verifyPulse(golden.pulse, { publicKey: golden.pulse.public_key }).valid).toBe(true);
  });

  it('is 260 bytes with fields at the documented offsets', () => {
    const p = signPulse({ index: 0x0102030405, timestamp_ms: 0x0a0b0c0d0e0f });
    const m = pulseMessage(p);
    expect(m).not.toBeNull();
    const bytes = m as Uint8Array;
    expect(bytes.length).toBe(MESSAGE_BYTES);
    const slice = ([a, b]: readonly [number, number]) => hex(bytes.slice(a, b));
    expect(Buffer.from(bytes.slice(...MESSAGE_LAYOUT.domain)).toString('ascii')).toBe('PUFFER-BEACON-V1');
    expect(slice(MESSAGE_LAYOUT.index)).toBe('0000000102030405');
    expect(slice(MESSAGE_LAYOUT.timestampMs)).toBe('00000a0b0c0d0e0f');
    expect(slice(MESSAGE_LAYOUT.periodMs)).toBe('0000ea60');
    expect(slice(MESSAGE_LAYOUT.localRandom)).toBe(p.local_random);
    expect(slice(MESSAGE_LAYOUT.precommitment)).toBe(p.precommitment);
    expect(slice(MESSAGE_LAYOUT.prevOutput)).toBe(p.prev_output);
    expect(slice(MESSAGE_LAYOUT.summaryHash)).toBe(p.summary_hash);
    expect(hex(bytes)).toBe(hex(handMessage(p)));
  });

  it('encodes u64 fields above 2^32 exactly', () => {
    const m = encodePulseMessage({
      index: Number.MAX_SAFE_INTEGER,
      timestampMs: 2 ** 40 + 1,
      periodMs: 0xffffffff,
      localRandom: new Uint8Array(64),
      precommitment: new Uint8Array(64),
      prevOutput: new Uint8Array(64),
      summaryHash: new Uint8Array(32),
    });
    expect(hex(m.slice(16, 24))).toBe('001fffffffffffff');
    expect(hex(m.slice(24, 32))).toBe('0000010000000001');
    expect(hex(m.slice(32, 36))).toBe('ffffffff');
  });

  it('rejects malformed fields instead of encoding garbage', () => {
    const base = {
      index: 1,
      timestampMs: 1,
      periodMs: 1,
      localRandom: new Uint8Array(64),
      precommitment: new Uint8Array(64),
      prevOutput: new Uint8Array(64),
      summaryHash: new Uint8Array(32),
    };
    expect(() => encodePulseMessage({ ...base, index: -1 })).toThrow(RangeError);
    expect(() => encodePulseMessage({ ...base, periodMs: 2 ** 32 })).toThrow(RangeError);
    expect(() => encodePulseMessage({ ...base, summaryHash: new Uint8Array(31) })).toThrow(RangeError);
  });
});

describe('verifyPulse', () => {
  const chain = honestChain(4);
  const [genesis, second, third] = chain as [BeaconPulse, BeaconPulse, BeaconPulse];

  it('accepts an honest chain with every applicable check passing', () => {
    const result = verifyChain(chain, { publicKey: PUBLIC });
    expect(result.valid).toBe(true);
    expect(result.firstFailure).toBeNull();
    expect(result.results[0]?.checks.genesis).toBe('pass');
    expect(result.results[0]?.checks.link).toBe('skip');
    const later = result.results[2];
    expect(later?.checks).toEqual({
      public_key: 'pass',
      summary_hash: 'pass',
      signature: 'pass',
      output: 'pass',
      index: 'pass',
      timestamp: 'pass',
      link: 'pass',
      precommitment: 'pass',
      genesis: 'skip',
    });
  });

  it('checks a lone, unpinned pulse for internal consistency only', () => {
    const result = verifyPulse(second);
    expect(result.valid).toBe(true);
    expect(result.checks.public_key).toBe('skip');
    expect(result.checks.signature).toBe('pass');
  });

  it('trusts the previous pulse key when none is pinned, so a chain cannot switch keys', () => {
    expect(verifyPulse(second, { prev: genesis }).checks.public_key).toBe('pass');
    const switched = signPulse({ index: 1, prev_output: genesis.output, secret: OTHER_SECRET });
    expect(verifyPulse(switched).valid).toBe(true); // self-consistent on its own...
    expect(verifyPulse(switched, { prev: genesis }).errors).toEqual(['wrong_public_key', 'bad_signature']); // ...but not a continuation of this chain
  });

  it('accepts uppercase hex (same bytes)', () => {
    const upper = { ...second, signature: second.signature.toUpperCase() };
    expect(verifyPulse(upper, { prev: genesis, publicKey: PUBLIC.toUpperCase() }).valid).toBe(true);
  });

  const cases: [string, () => [BeaconPulse, BeaconPulse | null, string | null], string][] = [
    ['a pulse naming a different key', () => [second, genesis, OTHER_PUBLIC], 'wrong_public_key'],
    [
      'a forgery signed by another key',
      () => [signPulse({ index: 1, prev_output: genesis.output, secret: OTHER_SECRET }), genesis, PUBLIC],
      'wrong_public_key',
    ],
    [
      'an edited summary',
      () => [{ ...second, summary: '{"fish_count":99}' }, genesis, PUBLIC],
      'bad_summary_hash',
    ],
    [
      'a flipped signature bit',
      () => [{ ...second, signature: flipHexByte(second.signature) }, genesis, PUBLIC],
      'bad_signature',
    ],
    [
      'an edited local_random',
      () => [{ ...second, local_random: flipHexByte(second.local_random) }, genesis, PUBLIC],
      'bad_signature',
    ],
    [
      'a summary re-hashed without re-signing',
      () => {
        const summary = '{"fish_count":99}';
        return [
          { ...second, summary, summary_hash: toHex(b2b(new TextEncoder().encode(summary), 32)) },
          genesis,
          PUBLIC,
        ];
      },
      'bad_signature',
    ],
    [
      'a replaced output',
      () => [{ ...second, output: flipHexByte(second.output, 63) }, genesis, PUBLIC],
      'bad_output',
    ],
    [
      'a skipped index',
      () => [
        signPulse({ index: 2, prev_output: genesis.output, local_random: randomFor(1) }),
        genesis,
        PUBLIC,
      ],
      'bad_index',
    ],
    [
      'a timestamp that does not advance',
      () => [
        signPulse({ index: 2, prev_output: second.output, timestamp_ms: second.timestamp_ms }),
        second,
        PUBLIC,
      ],
      'bad_timestamp',
    ],
    [
      'a pulse linked to the wrong predecessor',
      () => [signPulse({ index: 2, prev_output: genesis.output }), second, PUBLIC],
      'broken_link',
    ],
    [
      'a local_random that breaks the precommitment',
      () => [
        signPulse({ index: 2, prev_output: second.output, local_random: randomFor(77) }),
        second,
        PUBLIC,
      ],
      'bad_precommitment',
    ],
    [
      'a genesis pulse with a non-zero prev_output',
      () => [signPulse({ index: 0, prev_output: 'ab'.repeat(64) }), null, PUBLIC],
      'bad_genesis',
    ],
  ];

  it.each(cases)('rejects %s', (_name, build, expected) => {
    const [pulse, prev, key] = build();
    const result = verifyPulse(pulse, { prev, publicKey: key });
    expect(result.valid).toBe(false);
    expect(result.error).toBe(expected);
    expect(result.errors).toContain(expected);
  });

  it('reports every failing check, in the canonical order', () => {
    const tampered = { ...third, signature: flipHexByte(third.signature), prev_output: genesis.output };
    const result = verifyPulse(tampered, { prev: second, publicKey: PUBLIC });
    expect(result.errors).toEqual(['bad_signature', 'bad_output', 'broken_link']);
    const order = result.errors.map((e) => BEACON_ERRORS.indexOf(e));
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });

  it('turns malformed fields into failed checks, never exceptions', () => {
    const broken: BeaconPulse = {
      ...second,
      signature: 'zz',
      output: '',
      summary_hash: 'abc',
      public_key: 'not-hex',
      prev_output: '00',
    };
    const result = verifyPulse(broken, { prev: genesis, publicKey: 'also-not-hex' });
    expect(result.errors).toEqual([
      'wrong_public_key',
      'bad_summary_hash',
      'bad_signature',
      'bad_output',
      'broken_link',
    ]);
  });

  it('flags a chain from its first broken pulse', () => {
    const forged = [
      genesis,
      second,
      { ...third, output: flipHexByte(third.output) },
      chain[3] as BeaconPulse,
    ];
    const result = verifyChain(forged, { publicKey: PUBLIC });
    expect(result.valid).toBe(false);
    expect(result.firstFailure).toEqual({ position: 2, index: 2, error: 'bad_output' });
    // The successor commits to the *real* output, so the forged one breaks the link too.
    expect(result.results[3]?.error).toBe('broken_link');
  });
});

describe('mock beacon chain', () => {
  it('produces Python-compatible canonical summaries', () => {
    const golden = JSON.parse(
      readFileSync(new URL('./fixtures/golden-pulse.json', import.meta.url), 'utf8'),
    ) as { pulse: BeaconPulse };
    expect(canonicalJson(JSON.parse(golden.pulse.summary))).toBe(golden.pulse.summary);
    expect(canonicalJson({ name: 'Señor Bubbles', b: [true, null] })).toBe(
      '{"b":[true,null],"name":"Se\\u00f1or Bubbles"}',
    );
    expect(() => canonicalJson({ bits: 1.5 })).toThrow(TypeError);
  });

  it('signs a chain the browser verifier accepts', () => {
    let counter = 0;
    const chain = new BeaconChain({
      seed: seedFromLabel('test'),
      periodMs: 10_000,
      randomBytes: (n) => b2b(new TextEncoder().encode(`r${counter++}`), n),
    });
    for (let i = 0; i < 5; i++) {
      chain.append({ fish_count: i, mode: 'day', top_fish: [{ name: 'Nemo', bits: i }] }, 1_000 + i * 10_000);
    }
    const result = verifyChain(chain.pulses, { publicKey: chain.publicKeyHex });
    expect(result.firstFailure).toBeNull();
    expect(chain.latest?.index).toBe(4);
  });
});
