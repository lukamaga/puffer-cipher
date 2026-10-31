/**
 * A small, deterministic aquarium simulation that speaks the puffer wire
 * protocol, for developing and screenshotting the dashboard without the
 * Python backend.
 *
 * It models just enough of the real engine to make every panel behave
 * honestly: fish with persistent identities and track ids that change when a
 * fish hides and reappears, credit computed with the DESIGN 4.3 formula
 * (`samples * min(H, cap) * safety`, halved when DEGRADED), pools that
 * saturate at their capacity and reset on every draw, a Manchester LED
 * challenge with rare observation errors and the exact binomial confidence
 * formula, and receipts/beacon summaries derived from real draws.
 *
 * Every public method returns a type from `src/lib/protocol.ts`, so the
 * compiler keeps the mock and the dashboard on the same contract. Motion
 * uses a seeded PRNG (stable screenshots); the public `bits` display stream
 * uses `crypto.getRandomValues` like the real one.
 */

import type {
  BitsData,
  ChannelHealth,
  EntropyData,
  EntropyReceipt,
  FishCard,
  FrameData,
  HealthData,
  HealthState,
  LeaderboardEntry,
  LightMode,
  LivenessStatus,
  Point,
  StatusResponse,
} from '../src/lib/protocol.ts';
import type { JsonValue } from './beacon-chain.ts';

export interface AquariumOptions {
  width: number;
  height: number;
  fps: number;
  seed: number;
  /** `cycle`: 4 min of day, then 1 min of night (IR), repeating from t = 0. */
  daylight: LightMode | 'cycle';
}

const SENSOR_SITES = 4096;
const CAPACITY_BITS = 4096;
const SAFETY = 0.5;
const H_CAP = { sensor: 4.0, fish: 3.0 } as const;
const LED_WINDOW = 64;
const LED_LAGS = 6;
const TRAIL_LENGTH = 26;
const CYCLE_S = 300;
const DAY_S = 240;

const ROSTER = [
  { id: 'f1', name: 'Nemo', color: '#ff8c42', length: 58, girth: 30, cruise: 46, chaos: 0.62 },
  { id: 'f2', name: 'Dory', color: '#4d9df7', length: 66, girth: 34, cruise: 58, chaos: 0.88 },
  { id: 'f3', name: 'Bubbles', color: '#ffd23f', length: 44, girth: 28, cruise: 34, chaos: 0.47 },
  { id: 'f4', name: 'Gill', color: '#b388ff', length: 76, girth: 38, cruise: 28, chaos: 0.31 },
  { id: 'f5', name: 'Pearl', color: '#ff6f9c', length: 50, girth: 30, cruise: 40, chaos: 0.55 },
  { id: 'f6', name: 'Tako', color: '#2ed3b7', length: 54, girth: 28, cruise: 52, chaos: 0.74 },
] as const;

type Rng = () => number;

/** mulberry32: tiny, fast, good enough for animation (never for secrets). */
function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const mod = (a: number, n: number) => ((a % n) + n) % n;

/** P[Binomial(n, 1/2) >= k], exact for n <= 64 in doubles. */
function binomialTail(n: number, k: number): number {
  let coefficient = 1;
  let total = 0;
  for (let i = 0; i <= n; i++) {
    if (i >= k) total += coefficient;
    coefficient = (coefficient * (n - i)) / (i + 1);
  }
  return total / 2 ** n;
}

interface Fish {
  readonly id: string;
  readonly name: string;
  readonly color: string;
  readonly length: number;
  readonly girth: number;
  readonly cruise: number;
  readonly personality: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  heading: number;
  dartUntil: number;
  hiddenUntil: number;
  trackId: number;
  identifiedAt: number;
  trail: Point[];
  turnRate: number;
  accel: number;
  speedAvg: number;
  speedMax: number;
  activity: number;
  chaos: number;
  bitsTotal: number;
  bitsToday: number;
  bitsWeek: number;
  distance: number;
  seen: number;
  /** Credit attributed to this fish since the last draw. */
  pool: number;
}

export class MockAquarium {
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  private readonly rng: Rng;
  private readonly daylight: AquariumOptions['daylight'];
  private readonly fish: Fish[];
  private nextTrackId = 1;

  /** Simulated monotonic clock (seconds) and frame counter. */
  t = 0;
  frameIndex = 0;

  private sensorH = 3.3;
  private fishH = 2.6;
  private rate = { sensor: 0, fish: 0, system: 0 };
  private credited = { sensor: 0, fish: 0, system: 0 };
  private pool = { sensor: 0, fish: 0 };
  private poolSinceFrame = 0;
  private drawTimes: number[] = [];

  private ledCommanded: number[] = [];
  private ledObserved: number[] = [];
  private ledLag = 0.14;

  constructor(options: AquariumOptions) {
    this.width = options.width;
    this.height = options.height;
    this.fps = options.fps;
    this.daylight = options.daylight;
    this.rng = mulberry32(options.seed);
    this.fish = ROSTER.map((r, i) => {
      const heading = this.rng() * 2 * Math.PI;
      return {
        ...r,
        personality: r.chaos,
        x: 80 + this.rng() * (this.width - 160),
        y: 70 + this.rng() * (this.height - 140),
        vx: Math.cos(heading) * r.cruise,
        vy: Math.sin(heading) * r.cruise * 0.5,
        heading,
        dartUntil: 0,
        hiddenUntil: 0,
        trackId: this.nextTrackId++,
        identifiedAt: 0,
        trail: [],
        turnRate: 0,
        accel: 0,
        speedAvg: r.cruise,
        speedMax: r.cruise * 2,
        activity: 0.7,
        chaos: r.chaos,
        bitsTotal: 2.1e6 + i * 3.3e5,
        bitsToday: 4.2e4 + i * 3.1e3,
        bitsWeek: 6.1e5 * (0.6 + r.chaos),
        distance: 3.9e6 * (0.5 + r.cruise / 60),
        seen: 86_400 * (6 + i * 0.4),
        pool: 0,
      };
    });
  }

  get mode(): LightMode {
    if (this.daylight !== 'cycle') return this.daylight;
    return mod(this.t, CYCLE_S) < DAY_S ? 'day' : 'night';
  }

  private visible(f: Fish): boolean {
    return this.t >= f.hiddenUntil;
  }

  private get presentCount(): number {
    return this.fish.filter((f) => this.visible(f)).length;
  }

  /** Advance one frame. */
  step(): void {
    const dt = 1 / this.fps;
    this.t += dt;
    this.frameIndex += 1;
    const night = this.mode === 'night';
    for (const f of this.fish) this.swim(f, dt, night);
    this.updateEntropy(dt, night);
    this.updateLed();
  }

  private swim(f: Fish, dt: number, night: boolean): void {
    const rng = this.rng;
    if (this.visible(f) && this.presentCount === ROSTER.length && rng() < 0.0035 * dt) {
      // Hide behind the rock: the tracker loses the fish and re-ID must find it again.
      f.hiddenUntil = this.t + 5 + rng() * 4;
    }
    const wasHidden = f.trail.length === 0;
    if (!this.visible(f)) {
      f.trail = [];
      return;
    }
    if (wasHidden && this.t > 1) {
      f.trackId = this.nextTrackId++;
      f.identifiedAt = this.t + 1.2;
    }

    const calm = night ? 0.45 : 1;
    if (rng() < 0.08 * dt * (0.4 + f.personality)) f.dartUntil = this.t + 0.4 + rng() * 0.5;
    const darting = this.t < f.dartUntil;
    const turnNoise = (0.9 + 1.6 * f.personality) * calm;
    let desired = f.heading + gaussian(rng) * turnNoise * Math.sqrt(dt);

    // Steer smoothly away from the glass.
    const margin = 70;
    const cx = this.width / 2;
    const cy = this.height / 2;
    const nearWall = f.x < margin || f.x > this.width - margin || f.y < margin || f.y > this.height - margin;
    if (nearWall) {
      const toCentre = Math.atan2(cy - f.y, cx - f.x);
      desired += wrapAngle(toCentre - desired) * 2.4 * dt;
    }
    const prevHeading = f.heading;
    f.heading = wrapAngle(desired);

    const speed = f.cruise * calm * (darting ? 2.8 : 1) * (0.85 + 0.3 * rng());
    const vx = Math.cos(f.heading) * speed;
    const vy = Math.sin(f.heading) * speed * 0.55;
    f.accel = Math.hypot(vx - f.vx, vy - f.vy) / dt;
    f.turnRate = wrapAngle(f.heading - prevHeading) / dt;
    f.vx = vx;
    f.vy = vy;
    f.x = clamp(f.x + vx * dt, f.length / 2, this.width - f.length / 2);
    f.y = clamp(f.y + vy * dt, f.girth / 2, this.height - f.girth / 2);
    f.trail.push([f.x, f.y]);
    if (f.trail.length > TRAIL_LENGTH) f.trail.shift();

    const moved = Math.hypot(vx, vy);
    f.distance += moved * dt;
    f.seen += dt;
    f.speedAvg += (moved - f.speedAvg) * 0.01;
    f.speedMax = Math.max(f.speedMax * 0.9995, moved);
    f.activity += ((moved > 25 ? 1 : 0) - f.activity) * 0.01;
    const instant = clamp((Math.abs(f.turnRate) / 4 + f.accel / 400) / 2, 0, 1);
    f.chaos += (0.6 * f.personality + 0.4 * instant - f.chaos) * 0.01;
  }

  private updateEntropy(dt: number, night: boolean): void {
    // H estimates refresh every 30 frames (DESIGN 4.3), as a slow random walk.
    if (this.frameIndex % 30 === 0) {
      const sensorTarget = night ? 3.85 : 3.3;
      const fishTarget = night ? 1.7 : 2.65;
      this.sensorH = clamp(
        this.sensorH + (sensorTarget - this.sensorH) * 0.3 + gaussian(this.rng) * 0.06,
        2.4,
        4.6,
      );
      this.fishH = clamp(this.fishH + (fishTarget - this.fishH) * 0.3 + gaussian(this.rng) * 0.08, 1.2, 3.4);
    }
    const sensorCredit = SENSOR_SITES * Math.min(this.sensorH, H_CAP.sensor) * SAFETY;
    const confirmed = this.fish.filter((f) => this.visible(f) && f.trail.length >= 3);
    const degraded = this.fishHealthState() === 'degraded';
    const perFish = Math.min(this.fishH, H_CAP.fish) * SAFETY * (degraded ? 0.5 : 1);
    const fishCredit = perFish * confirmed.length;

    this.credited.sensor += sensorCredit;
    this.credited.fish += fishCredit;
    const alpha = 0.08;
    this.rate.sensor += (sensorCredit / dt - this.rate.sensor) * alpha;
    this.rate.fish += (fishCredit / dt - this.rate.fish) * alpha;
    const recentDraws = this.drawTimes.filter((d) => this.t - d < 60).length;
    this.rate.system = (recentDraws * 256) / 60;

    this.pool.sensor = Math.min(CAPACITY_BITS, this.pool.sensor + sensorCredit);
    const room = CAPACITY_BITS - this.pool.fish;
    const accepted = Math.min(room, fishCredit);
    this.pool.fish += accepted;
    for (const f of confirmed) {
      const share = fishCredit > 0 ? (accepted * perFish) / fishCredit : 0;
      f.pool += share;
      f.bitsTotal += perFish;
      f.bitsToday += perFish;
      f.bitsWeek += perFish;
    }
  }

  private updateLed(): void {
    // symbol_s = 0.2 s, Manchester: one half-symbol per 0.1 s frame. Bit 1 = on->off.
    if (this.frameIndex % 2 === 0) {
      const bit = this.rng() < 0.5 ? 1 : 0;
      this.ledCommanded.push(bit, 1 - bit);
      for (const half of [bit, 1 - bit]) this.ledObserved.push(this.rng() < 0.012 ? 1 - half : half);
    }
    const keep = LED_WINDOW + 8;
    if (this.ledCommanded.length > keep) {
      this.ledCommanded.splice(0, this.ledCommanded.length - keep);
      this.ledObserved.splice(0, this.ledObserved.length - keep);
    }
    this.ledLag = clamp(this.ledLag + gaussian(this.rng) * 0.002, 0.11, 0.18);
  }

  private fishHealthState(): HealthState {
    return this.presentCount < ROSTER.length ? 'degraded' : 'ok';
  }

  frame(): FrameData {
    return {
      index: this.frameIndex,
      t: this.t,
      width: this.width,
      height: this.height,
      jpeg: null,
      tracks: this.fish
        .filter((f) => this.visible(f) && f.trail.length > 0)
        .map((f) => {
          const identified = this.t >= f.identifiedAt;
          const w = f.length * (0.6 + 0.4 * Math.abs(Math.cos(f.heading)));
          const h = f.girth * (0.85 + 0.15 * Math.abs(Math.sin(f.heading)));
          return {
            track_id: f.trackId,
            fish_id: identified ? f.id : null,
            name: identified ? f.name : null,
            color: identified ? f.color : null,
            bbox: [f.x - w / 2, f.y - h / 2, f.x + w / 2, f.y + h / 2] as const,
            trail: [...f.trail],
            speed: Math.hypot(f.vx, f.vy),
            heading: f.heading,
          };
        }),
    };
  }

  entropy(): EntropyData {
    return {
      t: this.t,
      mode: this.mode,
      channels: {
        sensor: {
          bits_per_s: this.rate.sensor,
          h_per_sample: this.sensorH,
          credited_total: this.credited.sensor,
          health: 'ok',
        },
        fish: {
          bits_per_s: this.rate.fish,
          h_per_sample: this.fishH,
          credited_total: this.credited.fish,
          health: this.fishHealthState(),
        },
        system: {
          bits_per_s: this.rate.system,
          h_per_sample: null,
          credited_total: this.credited.system,
          health: 'ok',
        },
      },
      pool: { ...this.pool },
    };
  }

  bits(): BitsData {
    const bytes = crypto.getRandomValues(new Uint8Array(32));
    return { t: this.t, bits: Array.from(bytes, (b) => b.toString(2).padStart(8, '0')).join('') };
  }

  private channelHealth(): Record<'sensor' | 'fish', ChannelHealth> {
    const liveness = this.led().state === 'verified' ? 'ok' : 'failed';
    const common = {
      rct: 'ok',
      apt: 'ok',
      frozen_frame: 'ok',
      low_entropy: 'ok',
      source_timeout: 'ok',
      liveness,
    } as const;
    const fishState = this.fishHealthState();
    const missing = ROSTER.length - this.presentCount;
    return {
      sensor: { state: 'ok', reasons: [], checks: { ...common } },
      fish: {
        state: fishState,
        reasons:
          missing > 0
            ? [`fish_presence: ${this.presentCount} of ${ROSTER.length} expected fish visible; credit halved`]
            : [],
        checks: { ...common, fish_presence: fishState },
      },
    };
  }

  health(): HealthData {
    const channels = this.channelHealth();
    const overall = channels.fish.state === 'ok' && channels.sensor.state === 'ok' ? 'ok' : 'degraded';
    return { overall, channels };
  }

  led(): LivenessStatus {
    const n = Math.min(LED_WINDOW, this.ledCommanded.length);
    const expected = this.ledCommanded.slice(-n);
    const observed = this.ledObserved.slice(-n);
    const matches = expected.filter((b, i) => b === observed[i]).length;
    const ratio = n > 0 ? matches / n : 0;
    const confidence = n > 0 ? -Math.log2(binomialTail(n, matches) * LED_LAGS) : 0;
    return {
      state: n < LED_WINDOW ? 'calibrating' : ratio >= 0.9 ? 'verified' : 'failed',
      match_ratio: ratio,
      lag_s: this.ledLag,
      confidence_bits: Math.max(0, confidence),
      expected: expected.join(''),
      observed: observed.join(''),
      window_start: this.t - n * 0.1,
      window_end: this.t,
    };
  }

  fishCards(): FishCard[] {
    return this.fish.map((f) => ({
      fish_id: f.id,
      name: f.name,
      color: f.color,
      present: this.visible(f),
      track_id: this.visible(f) ? f.trackId : null,
      stats: {
        speed_avg: f.speedAvg,
        speed_max: f.speedMax,
        activity: f.activity,
        chaos_score: f.chaos,
        bits_total: f.bitsTotal,
        bits_today: f.bitsToday,
        distance_px: f.distance,
        seen_s: f.seen,
      },
    }));
  }

  leaderboard(days: number): LeaderboardEntry[] {
    const scale = Math.max(1, days) / 7;
    return this.fish
      .map((f) => ({
        fish_id: f.id,
        name: f.name,
        chaos_score: 0.7 * f.personality + 0.3 * f.chaos,
        bits: f.bitsWeek * scale,
      }))
      .sort((a, b) => b.chaos_score - a.chaos_score)
      .map((entry, i) => ({ rank: i + 1, ...entry }));
  }

  status(uptimeS: number): StatusResponse {
    const health = this.channelHealth();
    return {
      source: 'mock simulator',
      mode: this.mode,
      fps: this.fps,
      uptime_s: uptimeS,
      frames: this.frameIndex,
      pool: { ...this.pool },
      liveness: this.led(),
      health: {
        sensor: { channel: 'sensor', ...health.sensor },
        fish: { channel: 'fish', ...health.fish },
      },
    };
  }

  /** Consume the pools for one PufferRNG draw and describe it (DESIGN 4.4-4.5). */
  draw(purpose: string, bytes: number): EntropyReceipt {
    const camera = this.pool.sensor + this.pool.fish;
    const share = (bits: number) => (camera > 0 ? bits / camera : 0);
    const liveness = this.led();
    const health = this.channelHealth();
    const receipt: EntropyReceipt = {
      receipt_version: 1,
      purpose,
      bytes,
      timestamp_ms: Date.now(),
      frames: [this.poolSinceFrame, this.frameIndex],
      mode: this.mode,
      liveness: { state: liveness.state, confidence_bits: liveness.confidence_bits },
      channels: {
        sensor: {
          credited_bits: this.pool.sensor,
          share: share(this.pool.sensor),
          health: health.sensor.state,
        },
        fish: { credited_bits: this.pool.fish, share: share(this.pool.fish), health: health.fish.state },
        system: { credited_bits: 256, share: null, health: 'ok' },
      },
      fish: this.fish
        .filter((f) => f.pool > 0)
        .sort((a, b) => b.pool - a.pool)
        .map((f) => ({
          fish_id: f.id,
          name: f.name,
          bits: f.pool,
          share: this.pool.fish > 0 ? f.pool / this.pool.fish : 0,
        })),
    };
    this.pool = { sensor: 0, fish: 0 };
    for (const f of this.fish) f.pool = 0;
    this.poolSinceFrame = this.frameIndex + 1;
    this.credited.system += 256;
    this.drawTimes = [...this.drawTimes.filter((d) => this.t - d < 60), this.t];
    return receipt;
  }

  /** Beacon summary for a draw: integers only (DESIGN 3.4). */
  beaconSummary(receipt: EntropyReceipt): JsonValue {
    return {
      fish_count: this.presentCount,
      fish_bits: Math.round(receipt.channels.fish.credited_bits),
      sensor_bits: Math.round(receipt.channels.sensor.credited_bits),
      system_bits: 256,
      mode: receipt.mode,
      liveness: receipt.liveness.state,
      liveness_confidence_bits: Math.round(receipt.liveness.confidence_bits),
      health: { fish: receipt.channels.fish.health, sensor: receipt.channels.sensor.health },
      top_fish: receipt.fish
        .slice(0, 3)
        .map((f) => ({ fish_id: f.fish_id, name: f.name ?? f.fish_id, bits: Math.round(f.bits) })),
      frames: receipt.frames ? [...receipt.frames] : null,
    };
  }
}
