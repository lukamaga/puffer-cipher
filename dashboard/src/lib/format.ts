/**
 * Number and time formatting for the UI. Compact, unit-bearing strings so a
 * reader never has to count digits (`81.9 kbit/s`, not `81920.4`).
 */

const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 });
const grouped = new Intl.NumberFormat('en', { maximumFractionDigits: 0 });

/** 1,284 below ten thousand, 12.9K above. */
export function formatCount(value: number): string {
  return Math.abs(value) < 10_000 ? grouped.format(value) : compact.format(value);
}

const BIT_UNITS = ['bit', 'kbit', 'Mbit', 'Gbit'] as const;

/** Decimal SI bit quantities: 950 bit, 81.9 kbit, 1.2 Mbit. */
export function formatBits(bits: number, suffix = ''): string {
  let value = bits;
  let unit = 0;
  while (Math.abs(value) >= 1000 && unit < BIT_UNITS.length - 1) {
    value /= 1000;
    unit += 1;
  }
  const digits = unit === 0 || Math.abs(value) >= 100 ? 0 : 1;
  return `${value.toFixed(digits)} ${BIT_UNITS[unit]}${suffix}`;
}

export const formatRate = (bitsPerSecond: number): string => formatBits(bitsPerSecond, '/s');

/** 6d 04h, 3h 12m, 4m 05s, 42s. */
export function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const d = Math.floor(s / 86_400);
  if (d > 0) return `${d}d ${String(Math.floor((s % 86_400) / 3600)).padStart(2, '0')}h`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rest = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(rest).padStart(2, '0')}s`;
  return `${rest}s`;
}

/** Relative time label: "now", "-30s", "-2m", "-2m 37s". */
export function formatAgo(seconds: number): string {
  const s = Math.round(Math.abs(seconds));
  if (s === 0) return 'now';
  if (s < 60) return `-${s}s`;
  const rest = s % 60;
  return rest === 0 ? `-${s / 60}m` : `-${Math.floor(s / 60)}m ${String(rest).padStart(2, '0')}s`;
}

const clock = new Intl.DateTimeFormat('en-GB', {
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hour12: false,
});

export function formatClock(epochMs: number): string {
  return clock.format(new Date(epochMs));
}

/** Group long hex into readable blocks without changing it. */
export function groupHex(hex: string, size = 8): string {
  return hex.match(new RegExp(`.{1,${size}}`, 'g'))?.join(' ') ?? hex;
}

export function shortHex(hex: string, head = 8, tail = 6): string {
  return hex.length <= head + tail + 1 ? hex : `${hex.slice(0, head)}…${hex.slice(-tail)}`;
}
