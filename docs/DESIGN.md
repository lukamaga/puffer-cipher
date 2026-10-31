# puffer-cipher — design contract

This document is the single source of truth for module boundaries, byte
formats and wire protocols. Code, tests and the dashboard must agree with it.
Shared Python types live in `src/puffer/types.py`.

## 0. Non-negotiable rules

1. **No home-made cryptography.** Every cryptographic operation is a call into
   libsodium via PyNaCl (`nacl.bindings`, `nacl.signing`, `nacl.pwhash`,
   `nacl.utils`) or Python's `hashlib.blake2b` (the reference BLAKE2
   implementation). In the browser: `@noble/curves` (Ed25519) and
   `@noble/hashes` (BLAKE2b). We design *protocols and formats* on top of
   these primitives, never primitives.
2. **Never weaker than the OS.** Every secret output is
   `BLAKE2b(key = os.urandom(32), data = pool material)` expanded with
   libsodium's ChaCha20 deterministic generator. If every fish dies and the
   camera is replaced by an attacker's video, output is still exactly as
   strong as `os.urandom`. Fish only *add* entropy.
3. **Credit is earned, not assumed.** Entropy is credited only from
   well-defined sample sequences with a min-entropy estimate (SP 800-90B
   style), multiplied by a safety factor, and only while the channel passes
   its health tests *and* (when required) the LED liveness check.
4. **Honesty in reporting.** Statistical tests on hashed output prove nothing
   about entropy. Results always say which data (synthetic simulator,
   recorded video, live camera) produced them.

## 1. Data flow

```
              ┌──────────── raw frame bytes ───────────┐
camera/sim ──►│ SensorChannel  (temporal pixel noise)   ├──┐
  (Frame)     │ FishTracker → FishChannel (innovations) ├──┤ ChannelOutput
              └─────────────────────────────────────────┘  │
LED schedule ─► LivenessVerifier ──► LivenessStatus        │
                                         │                 ▼
                       HealthMonitor ◄───┴──── per-channel samples
                             │ (ok / startup / degraded / failed)
                             ▼
                       EntropyPool (BLAKE2b accumulators + credit ledger)
                             │ draw()
          os.urandom(32) ──► PufferRNG  ──► keys, nonces, salts, beacon values
                             │                         │
                         EntropyReceipt           XChaCha20-Poly1305
                      (per-channel + per-fish)     (libsodium)
```

## 2. Package ownership

| package | responsibility |
|---|---|
| `puffer.types` | shared dataclasses and protocols (this contract) |
| `puffer.crypto` | AEAD messages, encrypted file format, key files, Ed25519 keys |
| `puffer.entropy` | channels, health tests, pool, RNG, receipts, conditioned test streams |
| `puffer.entropy.estimators` | SP 800-90B min-entropy estimators |
| `puffer.analysis` | SP 800-22 suite, `ent`-equivalent, dieharder/ent wrappers, report builder |
| `puffer.sources` | simulator, video file, camera (OpenCV, picamera2), source spec parser |
| `puffer.hardware` | LED drivers (GPIO, simulated, null) |
| `puffer.vision` | detectors (motion, YOLO), Kalman, ByteTrack, re-identification, motion features |
| `puffer.liveness` | LED challenge schedule, LED detector/calibration, liveness verifier |
| `puffer.beacon` | pulse format, signing, chain verification, SQLite store, scheduler |
| `puffer.engine` | wires everything; produces snapshots and receipts |
| `puffer.fishstats` | fish character stats, attribution history, leaderboard |
| `puffer.api` | FastAPI REST + WebSocket |
| `puffer.cli` | `puffer` command line |
| `dashboard/` | React + Vite + TypeScript UI |

## 3. Cryptographic formats

All integers are big-endian. `BLAKE2b-n` means `hashlib.blake2b(digest_size=n/8)`.
Personalization strings (`person=`) are at most 16 bytes.

### 3.1 Key file (`*.key`, JSON, mode 0600)

```json
{
  "format": "puffer-key",
  "version": 1,
  "key_id": "a1b2c3d4e5f60718",
  "created": "2026-10-31T15:00:00Z",
  "key": "<base64url, no padding, 32 bytes>",
  "receipt": { "...": "EntropyReceipt JSON, optional, never secret" }
}
```

`key_id = hex(BLAKE2b-64(key, person=b"puffer-key-id"))` (16 hex chars).

### 3.2 Encrypted file (`*.puf`), libsodium secretstream XChaCha20-Poly1305

```
offset size  field
0      4     magic  b"PUFR"
4      1     version = 1
5      1     kdf: 0 = raw key (key file), 1 = Argon2id password
6      2     flags = 0 (reserved)
8      4     chunk_size (default 65536, 1 KiB .. 16 MiB)
12     8     key_id: first 8 bytes of BLAKE2b-64 key id (kdf=0); zeros (kdf=1)
-- kdf = 1 only --
20     8     opslimit (u64)
28     8     memlimit (u64)
36     16    salt (crypto_pwhash_SALTBYTES) from PufferRNG
-- always --
..     24    secretstream header (crypto_secretstream_xchacha20poly1305_HEADERBYTES)
..     ...   chunks
```

* `preamble` = every byte before the first chunk (magic .. secretstream header).
* Chunk *i* = `push(state, plaintext_i, ad = preamble if i == 0 else b"", tag)`
  where tag is `TAG_MESSAGE` for all but the last chunk and `TAG_FINAL` for the
  last. An empty file is one empty `TAG_FINAL` chunk.
* Each full chunk is `chunk_size + ABYTES` (17) bytes on disk; only the last
  may be shorter.
* Decryption **must** fail on: bad magic/version, unknown kdf, key-id
  mismatch (clear error before trying), any authentication failure, a stream
  that ends without `TAG_FINAL` (truncation), or bytes after `TAG_FINAL`.
* Output is written to a temp file and atomically renamed only after the
  final tag is verified, so a failed decryption never leaves partial
  plaintext.
* Password mode: `nacl.pwhash.argon2id.kdf(32, password, salt, opslimit,
  memlimit)`; defaults `OPSLIMIT_MODERATE`/`MEMLIMIT_MODERATE`, tests may use
  `*_MIN`.

### 3.3 Short AEAD messages (API demo, dashboard)

`crypto_aead_xchacha20poly1305_ietf_encrypt(plaintext, aad, nonce, key)` with
`aad = b"puffer-msg-v1"` and a 24-byte nonce from PufferRNG. Wire format:
`base64url(nonce || ciphertext)`.

### 3.4 Beacon pulse (Ed25519 + BLAKE2b hash chain)

```
M = b"PUFFER-BEACON-V1"   (16 bytes, ASCII)
  || u64(index)
  || u64(timestamp_ms)           unix epoch milliseconds
  || u32(period_ms)
  || local_random    (64)        fresh PufferRNG output for this pulse (revealed now)
  || precommitment   (64)        BLAKE2b-512(local_random of pulse index+1)
  || prev_output     (64)        output of pulse index-1; 64 zero bytes for index 0
  || summary_hash    (32)        BLAKE2b-256(summary_json_utf8)
len(M) = 260
signature = Ed25519_sign(signing_key, M)                       (64 bytes)
output    = BLAKE2b-512(M || signature)                         (64 bytes)
```

Verification of pulse *i* (all must hold):

1. `summary_hash == BLAKE2b-256(summary.encode("utf-8"))` (hash the exact string).
2. `Ed25519_verify(public_key, M, signature)`.
3. `output == BLAKE2b-512(M || signature)`.
4. Chain (if pulse *i-1* is known): `index_i == index_{i-1} + 1`,
   `timestamp_i > timestamp_{i-1}`, `prev_output_i == output_{i-1}`,
   `BLAKE2b-512(local_random_i) == precommitment_{i-1}`.
5. `index == 0` ⇒ `prev_output` is all zeros.

JSON form (hex is lowercase):

```json
{
  "version": 1,
  "index": 42,
  "timestamp_ms": 1793454000000,
  "period_ms": 60000,
  "local_random": "<128 hex>",
  "precommitment": "<128 hex>",
  "prev_output": "<128 hex>",
  "summary": "{\"fish_bits\":812,...}",
  "summary_hash": "<64 hex>",
  "signature": "<128 hex>",
  "output": "<128 hex>",
  "public_key": "<64 hex>"
}
```

`summary` is produced with `json.dumps(obj, sort_keys=True,
separators=(",", ":"), ensure_ascii=True)` and contains only ints, strings,
bools, lists and objects (no floats - entropy amounts are integer bits).
Fields: `fish_count`, `fish_bits`, `sensor_bits`, `system_bits`, `mode`,
`liveness` (state), `liveness_confidence_bits`, `health` ({channel: state}),
`top_fish` ([{`fish_id`, `name`, `bits`}]), `frames` ([first, last]).

Beacon values are **public**. They must never be used as secret keys; the
beacon draws its own `local_random` from PufferRNG like any other consumer.

Test vectors: `tests/vectors/beacon_vectors.json` (generated by
`python -m puffer.beacon.vectors`, deterministic signing key from a fixed
seed) are consumed by both pytest and the dashboard's vitest suite.

## 4. Entropy subsystem

### 4.1 Channels

* **sensor** (`SensorChannel.process(frame) -> ChannelOutput`)
  * `absorb` = 16-byte header (u64 frame index, f64 timestamp) + the whole
    raw frame bytes (sensor noise, reflections, bubbles, caustics - all of it).
  * `samples` = temporal differences `(I_t - I_{t-1}) mod 256` of the green
    channel at `K` (default 4096) fixed sample sites chosen once at start
    (seeded from `os.urandom`), excluding the LED region of interest and a
    border margin. First frame produces no samples.
* **fish** (`FishChannel.process(result: TrackingResult) -> ChannelOutput`)
  * `absorb` = packed float64 of every observation's
    (track_id, cx, cy, vx, vy, heading, accel, turn_rate, innovation_x, innovation_y).
  * `samples` = one uint8 per confirmed observation:
    `((qx & 0xF) << 4) | (qy & 0xF)` where `q = round(innovation * 8)`
    (1/8-pixel quantisation). `sample_owner[i]` = that observation's fish id
    (or `f"track-{track_id}"` when unidentified).
* **system**: `os.urandom(32)` per draw; displayed as 256 bits, always present.

### 4.2 Health (SP 800-90B section 4.4 + domain checks)

Per channel: Repetition Count Test (`C = 1 + ceil(-log2(alpha) / H)`),
Adaptive Proportion Test (`W = 512`, `C = 1 + CRITBINOM(W, 2^-H, 1 - alpha)`),
`alpha = 2^-20`, `H` = the channel's assessed per-sample min-entropy (a
config value, default sensor 2.0, fish 2.0). Start-up: the first 1024
samples must pass before credit begins (`STARTUP`). Domain checks:

| check | channel | rule (defaults) |
|---|---|---|
| `frozen_frame` | sensor (and fish) | identical frame hash, or mean abs diff < 0.05 for 15 consecutive frames ⇒ FAILED |
| `fish_presence` | fish | fewer than `min_fish` (1) confirmed tracks for 10 s ⇒ FAILED; fewer than expected ⇒ DEGRADED |
| `low_entropy` | each | rolling quick estimate below `min_bits_per_sample` (0.5) ⇒ FAILED |
| `source_timeout` | all camera | no frame for 5 s ⇒ FAILED |
| `liveness` | all camera | liveness required and not VERIFIED ⇒ FAILED |

FAILED ⇒ the channel's output is neither absorbed nor credited. Recovery
requires a cooldown (5 s) and a fresh start-up window.

### 4.3 Credit

`credit_bits = n_samples * min(H_est, H_cap) * safety_factor`

`H_est` is the rolling quick min-entropy estimate (MCV-based over the last
4096 samples, refreshed every 30 frames), `H_cap` caps per-sample credit
(sensor 4.0, fish 3.0), `safety_factor = 0.5`. DEGRADED halves credit again.
Fish credit is split between fish proportionally to their sample counts.

### 4.4 Pool and RNG

* `EntropyPool.absorb(output: ChannelOutput, credit_bits: float)` updates a
  per-channel BLAKE2b-512 accumulator (`person=b"puffer-pool-" + name`) and
  credit counters (per channel, per fish). Credit per channel saturates at
  `capacity_bits` (default 4096).
* `EntropyPool.draw() -> PoolDraw` returns the accumulator digests
  (`material`), the consumed credits, per-fish attribution, frame range and
  resets credits. Each accumulator is re-seeded with its own previous digest
  (`BLAKE2b(prev_digest, person=b"puffer-ratchet")`) so un-credited entropy
  carries over and past draws cannot be recomputed from future state.
* `PufferRNG.generate(n, purpose) -> (bytes, EntropyReceipt)`:
  `seed = BLAKE2b-256(key=os.urandom(32), person=b"puffer-seed-v1",
  data = u32(len(purpose)) || purpose || u64(counter) || draw.material)`;
  `out = nacl.utils.randombytes_deterministic(n, seed)`.
  `generate_key()` = `generate(32, "key")`, `nonce()` = `generate(24, "nonce")`.
* `PufferRNG.wait_for_credit(bits, timeout)` lets callers (CLI keygen) wait
  until the camera channels have contributed at least `bits` credited bits.

### 4.5 Receipt (`EntropyReceipt.to_dict()`)

```json
{
  "receipt_version": 1,
  "purpose": "key",
  "bytes": 32,
  "timestamp_ms": 1793454000000,
  "frames": [1200, 1290],
  "mode": "day",
  "liveness": {"state": "verified", "confidence_bits": 54.2},
  "channels": {
    "sensor": {"credited_bits": 812.0, "share": 0.61, "health": "ok"},
    "fish":   {"credited_bits": 512.5, "share": 0.39, "health": "ok"},
    "system": {"credited_bits": 256.0, "share": null, "health": "ok"}
  },
  "fish": [{"fish_id": "f3", "name": "Nemo", "bits": 120.5, "share": 0.31}]
}
```

`share` is over camera channels only (system is always 256 bits on top).

### 4.6 Conditioned test streams (analysis only)

`ConditionedStream(channel)` turns raw samples into test output **without**
system randomness: every block of samples whose credited min-entropy reaches
512 bits is compressed to `BLAKE2b-256(block, person=b"puffer-test-cond")`.
Used only to compare "fish only / sensor only / combined" in reports.

## 5. Liveness (anti-replay LED)

* `ChallengeSchedule(symbol_s=0.2, encoding="manchester")`: an endless stream of
  fresh random bits from `os.urandom`, generated lazily and remembered per
  symbol index; Manchester-encoded (bit 1 = on→off, bit 0 = off→on, each half
  `symbol_s / 2`). `state_at(t)` implements `types.LedSchedule`.
* LED drivers (`puffer.hardware.led`): `GpioLed(pin)` (gpiozero),
  `SimulatedLed` (records commanded states; the simulator renders it),
  `NullLed`. `LedController(schedule, driver)` drives a real LED from a thread.
* `LedDetector`: brightness in a region of interest (configured or found by
  `calibrate_roi`, which correlates candidate regions with the commanded
  pattern), adaptive two-cluster threshold, one on/off observation per frame.
* `LivenessVerifier.process(frame) -> LivenessStatus`: sample the observed
  state, compare with `schedule.state_at(t - lag)` for lags in
  `[0, max_lag_s]` (default 0.5 s), pick the best lag, and over a sliding
  window of 64 half-symbols require match ratio ≥ 0.9.
  `confidence_bits = -log2(P[Binomial(n, 0.5) ≥ matches] * n_lags)`.
  States follow `types.LivenessState`.
* A replayed recording shows an old pattern: P(match) ≈ 2^-confidence.

## 6. Vision

* `MotionDetector` (default, no weights): median background model with slow
  update + absolute difference + morphology + contours, area filter.
  `YoloDetector(weights)` (optional `ultralytics`, imported lazily).
* `ByteTracker`: ByteTrack (Zhang et al. 2022): constant-velocity Kalman in
  (cx, cy, a, h) space, high/low score two-stage IoU association with the
  Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), tentative →
  confirmed after 3 hits, lost buffer 30 frames. Exposes the measurement
  innovation for every update.
* `FishReID`: HSV colour-histogram appearance model; matches tracks to a
  persistent gallery of fish identities (`fish_id`, `name`); new stable fish
  are auto-enrolled with names from a built-in list.
* `FishTracker.process(frame) -> TrackingResult` composes detector, tracker,
  re-ID and motion features.

## 7. Wire protocol (FastAPI)

### 7.1 REST (prefix `/api`)

| method | path | response |
|---|---|---|
| GET | `/status` | `{source, mode, fps, uptime_s, frames, pool:{channel: bits}, liveness: LivenessStatus, health: {channel: HealthReport}}` |
| GET | `/random?n=32&format=hex` | `{data, receipt}` (n ≤ 1024; format hex/base64) |
| POST | `/demo/encrypt` `{plaintext}` | `{ciphertext, key, key_id, receipt}` (base64url; DEMO ONLY: key travels to the browser) |
| POST | `/demo/decrypt` `{ciphertext, key}` | `{plaintext}` or 400 `{detail}` |
| GET | `/fish` | `[FishCard]` |
| GET | `/leaderboard?days=7` | `[{rank, fish_id, name, chaos_score, bits}]` |
| GET | `/beacon/public-key` | `{algorithm: "Ed25519", public_key}` |
| GET | `/beacon/latest` | pulse JSON (404 if none) |
| GET | `/beacon/pulses?start=0&limit=50` | `[pulse]` ascending index |
| GET | `/beacon/pulse/{index}` | pulse JSON |

`FishCard = {fish_id, name, color: "#rrggbb", present: bool, track_id, stats:
{speed_avg, speed_max, activity, chaos_score, bits_total, bits_today,
distance_px, seen_s}}`.

### 7.2 WebSocket `/ws/live`

JSON text messages `{"type": <type>, "data": {...}}`:

| type | rate | data |
|---|---|---|
| `hello` | on connect | `{version, source, width, height, fps, beacon_public_key}` |
| `frame` | ~10 Hz | `{index, t, width, height, jpeg: <base64>, tracks: [{track_id, fish_id, name, color, bbox: [x1,y1,x2,y2], trail: [[x,y],...], speed, heading}]}` |
| `entropy` | 2 Hz | `{t, mode, channels: {sensor|fish|system: {bits_per_s, h_per_sample, credited_total, health}}, pool: {sensor, fish}}` |
| `bits` | 5 Hz | `{t, bits: "0101…"}` (256 display bits from a dedicated public `display` stream, never used as keys) |
| `health` | 1 Hz | `{overall, channels: {name: {state, reasons, checks}}}` |
| `led` | 2 Hz | `LivenessStatus` as JSON (`state` string, floats, `expected`, `observed`) |
| `fish` | 0.2 Hz | `{fish: [FishCard]}` |
| `beacon` | per pulse | pulse JSON |

## 8. Testing

* `pytest` everywhere; deterministic seeds; statistical tests marked `slow`.
* Crypto: round trips, tamper/truncation/extension detection, wrong key,
  property tests (hypothesis).
* SP 800-22: worked examples from NIST SP 800-22 rev1a and the e-expansion
  reference p-values (Appendix B).
* SP 800-90B: estimators on distributions with known min-entropy.
* Beacon: cross-language vectors shared with the dashboard.
