/**
 * Rolling time series for the entropy chart.
 *
 * Kept outside React state: `entropy` messages arrive at 2 Hz and the chart
 * needs a few minutes of them, so an append-only array trimmed by time is
 * cheaper than copying state on every message. Subscribers are told *that*
 * something changed and read a snapshot; `version` lets React memoise.
 *
 * Timestamps are the server's `t` (a monotonic clock, not wall time), so the
 * chart's x-axis is always relative to the newest sample. If `t` jumps
 * backwards the backend restarted and the old series is meaningless: it is
 * discarded rather than drawn as a line back in time.
 */

import type { ChannelName, EntropyData, LightMode } from './protocol.ts';

export interface EntropySample {
  t: number;
  mode: LightMode;
  rate: Record<ChannelName, number>;
}

export class EntropyHistory {
  readonly windowSeconds: number;
  private samples: EntropySample[] = [];
  private listeners = new Set<() => void>();
  private versionCounter = 0;

  constructor(windowSeconds = 300) {
    this.windowSeconds = windowSeconds;
  }

  get version(): number {
    return this.versionCounter;
  }

  get points(): readonly EntropySample[] {
    return this.samples;
  }

  push(data: EntropyData): void {
    const last = this.samples.at(-1);
    if (last && data.t <= last.t) {
      if (data.t < last.t) this.samples = [];
      else return; // duplicate delivery of the same instant
    }
    this.samples.push({
      t: data.t,
      mode: data.mode,
      rate: {
        sensor: data.channels.sensor.bits_per_s,
        fish: data.channels.fish.bits_per_s,
        system: data.channels.system.bits_per_s,
      },
    });
    const cutoff = data.t - this.windowSeconds;
    const firstKept = this.samples.findIndex((s) => s.t >= cutoff);
    if (firstKept > 0) this.samples.splice(0, firstKept);
    this.versionCounter += 1;
    for (const listener of this.listeners) listener();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
