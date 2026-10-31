/**
 * Display metadata for the entropy channels, shared by every panel so a
 * channel has one name and one colour across the whole dashboard (colour
 * follows the entity: sensor is always blue, fish always orange).
 */

import type { ChannelName } from './protocol.ts';

export const CHANNEL_META: Record<ChannelName, { label: string; color: string; source: string }> = {
  sensor: { label: 'Sensor noise', color: 'var(--series-sensor)', source: 'temporal pixel noise' },
  fish: { label: 'Fish motion', color: 'var(--series-fish)', source: 'Kalman innovations' },
  system: { label: 'System', color: 'var(--series-system)', source: 'os.urandom' },
};

/** DESIGN 4.4 default `capacity_bits`; the protocol does not carry it (yet). */
export const POOL_CAPACITY_BITS = 4096;
