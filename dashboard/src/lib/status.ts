/**
 * One visual vocabulary for every state in the system.
 *
 * Health states, liveness states and beacon check outcomes all map onto the
 * four fixed status tones (plus neutral). A tone is never shown by colour
 * alone: badges always carry an icon and a text label, so the dashboard
 * reads correctly for colour-blind users and in grayscale.
 */

import type { CheckOutcome } from './beacon.ts';
import type { HealthState, LivenessState } from './protocol.ts';

export type Tone = 'good' | 'warning' | 'serious' | 'critical' | 'neutral';
export type IconName = 'check' | 'clock' | 'alert' | 'x' | 'minus';

export interface StatusLook {
  tone: Tone;
  icon: IconName;
  label: string;
}

export const TONE_ICON: Record<Tone, IconName> = {
  good: 'check',
  warning: 'clock',
  serious: 'alert',
  critical: 'x',
  neutral: 'minus',
};

const look = (tone: Tone, label: string): StatusLook => ({ tone, icon: TONE_ICON[tone], label });

export const HEALTH_LOOK: Record<HealthState, StatusLook> = {
  ok: look('good', 'Healthy'),
  startup: look('warning', 'Start-up'),
  degraded: look('serious', 'Degraded'),
  failed: look('critical', 'Failed'),
};

export const LIVENESS_LOOK: Record<LivenessState, StatusLook> = {
  verified: look('good', 'Verified'),
  calibrating: look('warning', 'Calibrating'),
  failed: look('critical', 'Failed'),
  disabled: look('neutral', 'Disabled'),
};

export const CHECK_LOOK: Record<CheckOutcome, StatusLook> = {
  pass: look('good', 'pass'),
  fail: look('critical', 'fail'),
  skip: look('neutral', 'n/a'),
};

const CHECK_NAMES: Record<string, string> = {
  rct: 'Repetition count',
  repetition_count: 'Repetition count',
  apt: 'Adaptive proportion',
  adaptive_proportion: 'Adaptive proportion',
  startup: 'Start-up window',
  frozen_frame: 'Frozen frame',
  fish_presence: 'Fish presence',
  low_entropy: 'Min-entropy floor',
  source_timeout: 'Source timeout',
  liveness: 'LED liveness',
};

/** Human label for a health check name; unknown names are de-snake-cased. */
export function checkLabel(name: string): string {
  const known = CHECK_NAMES[name];
  if (known) return known;
  const words = name.replace(/[_-]+/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}
