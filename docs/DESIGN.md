# puffer-cipher design contract

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
  `*_MIN`. Text passwords are NFC-normalised and UTF-8 encoded.
* Header bounds checked on read *before* Argon2 runs (anti-DoS):
  `opslimit` in `[OPSLIMIT_MIN, 16]`, `memlimit` in `[MEMLIMIT_MIN, 1 GiB]`,
  `flags == 0`, password files carry an all-zero key id.
* Error classes: credential kind mismatch or key-id mismatch → `WrongKeyError`
  (before any decryption); bytes appended after a *short* final chunk merge
  into it and fail its tag (`AuthenticationError`), after a full-size final
  chunk → `FormatError`; EOF inside a chunk with fewer than `ABYTES` bytes or
  without `TAG_FINAL` → `TruncatedError`. Every case is rejected.
* The secretstream header (stream nonce) is generated inside libsodium's
  `init_push`; only the Argon2 salt comes from PufferRNG.

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
bools, lists and objects (no floats, no null - entropy amounts are integer
bits). Integers (`index`, `timestamp_ms`, summary values) are capped at
`2**53 - 1` so JavaScript's `JSON.parse` represents them exactly.

`timestamp_ms` is the start of the period slot containing the tick
(`now - now % period_ms`), at most one pulse per slot; after downtime indices
stay consecutive and timestamps skip missed slots.

Verification error codes, in check order: `wrong_public_key`,
`bad_summary_hash`, `bad_signature`, `bad_output`, `bad_genesis`,
`bad_index`, `bad_timestamp`, `broken_link`, `bad_precommitment`. The
signature is checked against the *trusted* key (explicit argument, else the
previous pulse's key, else trust-on-first-use of the embedded key).
Verifiers use strict RFC 8032 Ed25519 (`zip215: false` in @noble).
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
| `frozen_frame` | sensor (and fish) | frame bit-identical to any of the last 2048 frames (catches looped video), or mean abs diff < 0.05, for 15 consecutive frames ⇒ FAILED |
| `fish_presence` | fish | fewer than `min_fish` (1) confirmed tracks for 10 s ⇒ FAILED; fewer than expected ⇒ DEGRADED |
| `low_entropy` | each | rolling quick estimate below `min_bits_per_sample` (0.5) ⇒ FAILED |
| `source_timeout` | all camera | no frame for 5 s ⇒ FAILED |
| `liveness` | all camera | liveness required and not VERIFIED ⇒ FAILED |

FAILED ⇒ the channel's output is neither absorbed nor credited. Recovery
requires a cooldown (5 s) and a fresh start-up window.

### 4.3 Credit

`credit_bits = n_samples * min(H_est, H_cap) * safety_factor`

`H_est` is the rolling min-entropy estimate over the last 4096 samples
(chronological order), refreshed every 30 frames: the minimum of MCV, Lag
prediction and MultiMMC prediction on the bytes, and 8 x the Collision and
Compression estimates on their bitstring. The predictors catch periodic or
repeated data; the bitstring estimators catch structure inside each byte,
such as the sign bits of small pixel differences, `H_cap` caps per-sample credit
(sensor 4.0, fish 3.0), `safety_factor = 0.5`. DEGRADED halves credit again.
Fish credit is split between fish proportionally to their sample counts.

### 4.4 Pool and RNG

* `EntropyPool.absorb(output: ChannelOutput, credit_bits: float)` updates a
  per-channel BLAKE2b-512 accumulator (`person=(b"puffer-pool-" + name)[:16]`,
  each absorb framed as `u64(len) || bytes` so the encoding is injective) and
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
  `generate_key()` = `generate(32, "key")`.
* Public values (`nonce()`, `salt()`) use `generate(n, purpose, consume=False)`:
  `seed = BLAKE2b-256(key=os.urandom(32), person=b"puffer-public-v1", data =
  u32(len(purpose)) || purpose || u64(counter) || pool.snapshot_digest())`. They
  mix the current pool but consume no camera credit, so a nonce never starves the
  key generated next to it.
  Requests are expanded in 1 MiB chunks, one counter per chunk.
* Credit is a ledger of estimated *input* min-entropy, not output strength:
  every seed is 256 bits regardless of how much credit a draw consumed.
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

`share` is over camera channels only (system is always 256 bits on top);
it is `null` when no camera credit was drawn. A fish's `share` is its bits
divided by the fish channel's credited bits.

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
  `[0, max_lag_s]` (default 0.5 s, grid step = frame interval / 4), pick the
  best lag, and over a sliding window of the last 64 *bits* (128 chips; the
  two chips of a Manchester bit are complementary, so only bits count as
  independent evidence) require match ratio ≥ 0.9 and ≥ 24 confidence bits.
  `confidence_bits = -log2(P[Binomial(n, 0.5) ≥ matches] * n_lags)`.
  States follow `types.LivenessState`.
* A replayed recording shows an old pattern: P(match) ≈ 2^-confidence.
  `observed` uses `?` for bits that could not be decoded.
* Commitments: `BLAKE2b-256(person=b"puffer-led", kind || u64(first_symbol)
  || u32(n_bits) || packbits(bits))` with kind `C` (challenge) or `O`
  (observation, plus a decodability bitmap).
* Liveness proves *freshness* (the video is causally downstream of the
  challenge within `max_lag_s`), not authenticity: an attacker who watches
  the LED live and composites it into fake video passes. Rule 2 still holds.

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
| GET | `/entropy/history?seconds=300` | `[entropy message data]` ascending, from a server-side ring buffer (max 600 s) |

`FishCard.stats.activity` and `chaos_score` are fractions in 0..1.
`FishCard = {fish_id, name, color: "#rrggbb", present: bool, track_id, stats:
{speed_avg, speed_max, activity, chaos_score, bits_total, bits_today,
distance_px, seen_s}}`.

### 7.2 WebSocket `/ws/live`

JSON text messages `{"type": <type>, "data": {...}}`:

| type | rate | data |
|---|---|---|
| `hello` | on connect | `{version, source, width, height, fps, beacon_public_key}` |
| `frame` | ~10 Hz | `{index, t, width, height, jpeg: <base64>, tracks: [{track_id, fish_id, name, color, bbox: [x1,y1,x2,y2], trail: [[x,y],...], speed, heading}]}` |
| `entropy` | 2 Hz | `{t, mode, channels: {sensor|fish|system: {bits_per_s, h_per_sample, credited_total, health}}, pool: {sensor, fish, capacity_bits}}` (`h_per_sample` may be null for system) |
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
  reference p-values (Appendix B). Findings: NIST's `data.e` is the bits
  `10` followed by the fractional expansion; Appendix B's overlapping
  template (0.110434) and linear complexity (0.826335) values come from STS
  2.1.2's legacy probability tables, reproduced exactly with
  `probabilities=` (our default uses the rev1a tables).
* SP 800-90B: estimators on distributions with known min-entropy.
* Beacon: cross-language vectors shared with the dashboard.

## 9. Engine, configuration and runtime (phase 2)

### 9.1 `puffer.config.PufferConfig`

A dataclass loaded from an optional TOML file (`[puffer]` table) and
overridden by `PUFFER_<FIELD>` environment variables:

| field | default | meaning |
|---|---|---|
| `source` | `"synthetic"` | source spec (`puffer.sources.spec.open_source`) |
| `detector` | `"motion"` | `"motion"` or `"yolo:<weights.pt>"` |
| `liveness` | `"auto"` | `required` / `advisory` / `off` / `auto` (required for camera and synthetic, off for video files) |
| `led` | `"auto"` | `none`, `simulated`, `gpio:<pin>`; auto = simulated for synthetic, gpio:17 for camera on a Pi, none otherwise |
| `led_roi` | `None` | `(x, y, w, h)` or auto-calibration |
| `symbol_s`, `liveness_window_bits` | `0.2`, `64` | LED challenge timing |
| `sensor_sites` | `4096` | sensor sample sites |
| `safety_factor` | `0.5` | credit multiplier |
| `pool_capacity_bits` | `4096` | per-channel credit cap |
| `sensor_h_assessed`, `fish_h_assessed` | `2.0`, `2.0` | H used by RCT/APT cutoffs |
| `min_fish` | `1` | fish presence check |
| `beacon_enabled`, `beacon_period_s` | `True`, `60` | public beacon |
| `data_dir` | `./data` | SQLite DB (`puffer.db`), beacon signing key (`beacon.key`), re-ID gallery |
| `fish_names` | `{}` | fish_id -> display name overrides |
| `stream_fps`, `stream_width`, `jpeg_quality` | `10`, `640`, `70` | dashboard video stream |
| `host`, `port` | `127.0.0.1`, `8000` | API server |
| `static_dir` | `dashboard/dist` | built dashboard served at `/` when the directory exists |

### 9.2 `puffer.engine.EntropyEngine`

```python
@dataclass(frozen=True)
class ChannelStats:
    name: str
    bits_per_s: float          # credited bits per second (EMA)
    h_per_sample: float        # rolling min-entropy estimate
    credited_total: float      # since start
    health: HealthState

@dataclass(frozen=True)
class EngineSnapshot:
    frame_index: int; t: float; wall_time: float; fps: float
    mode: LightMode; width: int; height: int
    tracks: tuple[TrackSnapshot, ...]
    channels: dict[str, ChannelStats]      # sensor, fish, system
    pool: dict[str, float]                 # current credit per camera channel
    health: dict[str, HealthReport]; overall: HealthState
    liveness: LivenessStatus

class EntropyEngine:
    config: PufferConfig
    rng: PufferRNG                         # the ONLY way to get secret bytes
    def step(self) -> EngineSnapshot | None          # one frame; None when the source ends
    def collect(self, min_credit_bits: float, *, timeout_s: float, max_frames: int | None = None) -> bool
    def start(self) -> None                          # background thread (real-time sources)
    def stop(self) -> None
    def snapshot(self) -> EngineSnapshot | None
    def latest_frame(self) -> Frame | None
    def jpeg(self, *, max_width: int = 640, quality: int = 70) -> bytes | None
    def subscribe(self, callback: Callable[[EngineSnapshot], None]) -> Callable[[], None]
    def fish_cards(self) -> list[dict]               # DESIGN 7.1 FishCard
    def leaderboard(self, days: int = 7) -> list[dict]
    def receipt_context(self) -> ReceiptContext
    def close(self) -> None
```

Per frame (`step`): read frame → health.observe_frame → liveness.process →
tracker.process → sensor/fish channels → health samples + rolling estimate →
if usable: `pool.absorb(output, credit)` (else skipped) → fish stats →
snapshot → subscribers. Sensor sample sites exclude the LED region (the LED
pattern is our own challenge, not entropy). `rng.generate()` may be called
from any thread.

`puffer.runtime.build_engine(config, *, source=None) -> EntropyEngine` and
`build_beacon(config, engine) -> BeaconService` wire concrete components.
`puffer.fishstats.FishStats` persists per-fish daily aggregates (credited
bits, seconds seen, distance, chaos score) in `data_dir/puffer.db` and ranks
the leaderboard by mean chaos score.

### 9.3 CLI (`puffer`)

| command | behaviour |
|---|---|
| `keygen [-o fish.key] [--source] [--min-bits 256] [--timeout 60] [--json]` | collect camera entropy, write a key file, print the fish attribution |
| `encrypt FILE [-o OUT] [-k KEY \| --password] [--force]` | no key/password: make a fresh fish key and save it to `OUT.key` |
| `decrypt FILE [-o OUT] [-k KEY \| --password] [--force]` | default OUT strips `.puf` |
| `random [-n 32] [--format hex\|base64\|raw] [--stream]` | `--stream` writes raw bytes forever (for `dieharder -g 200`) |
| `simulate [-o demo.mp4] [--seconds 20] [--fish 6] [--mode] [--seed]` | render the aquarium simulator to a video |
| `serve [--config] [--source] [--host] [--port]` | engine + beacon + API + dashboard |
| `beacon verify FILE_OR_URL [--public-key]`, `beacon latest --url` | verify a chain offline |
| `analyze [--source] [--frames] [--out]` | statistics report (section 10) |
| `fish [--days 7]` | leaderboard |
| `info` | versions (libsodium, OpenCV), config, data dir |

## 10. Analysis report (`puffer.analysis.report`)

```python
@dataclass
class ReportConfig:
    sources: list[str]            # e.g. ["synthetic:seed=1,mode=day", "synthetic:seed=1,mode=night"]
    frames: int = 3000            # per source
    out_dir: Path = Path("docs/results")
    media_dir: Path = Path("docs/media")
    nist_sequences: int = 10      # 1e6-bit sequences per stream where available
    dieharder: bool = False       # run dieharder on the final PufferRNG output
    seed: int | None = None
def run_report(config: ReportConfig, *, progress: Callable[[str], None] | None = None) -> Path  # path of RESULTS.md
```


Collects raw per-channel samples from a source (simulator day, simulator
night, or a video file), runs the SP 800-90B assessment per channel, builds
conditioned streams *sensor only*, *fish only*, *combined* (no OS randomness)
and the final PufferRNG output (with OS randomness), runs SP 800-22
(multi-sequence proportion + uniformity), the `ent` equivalent, and dieharder
when installed (`puffer random --stream | dieharder -g 200`), and writes
`docs/results/RESULTS.md` + PNG plots to `docs/media/`. Every table states
the data source.
