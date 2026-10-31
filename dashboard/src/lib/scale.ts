/** Minimal scale helpers for hand-built SVG charts (no charting library). */

/** Smallest "nice" number (1, 2, 2.5, 5 x 10^k) that is >= value. */
export function niceCeil(value: number): number {
  if (!(value > 0)) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (step * magnitude >= value) return step * magnitude;
  }
  return 10 * magnitude;
}

export function linear(
  [d0, d1]: readonly [number, number],
  [r0, r1]: readonly [number, number],
): (v: number) => number {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

/** Axis tick: full digits under 10,000, compact above (0 / 20K / 80K). */
export function tickLabel(value: number, axisMax: number): string {
  if (axisMax < 10_000) return Math.round(value).toLocaleString('en');
  return new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

/** Index of the sample whose key is closest to `target` (keys ascending). */
export function nearestIndex<T>(items: readonly T[], target: number, key: (item: T) => number): number {
  let lo = 0;
  let hi = items.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (key(items[mid] as T) < target) lo = mid + 1;
    else hi = mid;
  }
  const prev = lo - 1;
  if (prev >= 0 && Math.abs(key(items[prev] as T) - target) <= Math.abs(key(items[lo] as T) - target)) {
    return prev;
  }
  return lo;
}
