import { type KeyboardEvent, type PointerEvent, useMemo, useState, useSyncExternalStore } from 'react';

import { CHANNEL_META, POOL_CAPACITY_BITS } from '../lib/channels.ts';
import { formatAgo, formatBits, formatRate } from '../lib/format.ts';
import type { EntropyHistory, EntropySample } from '../lib/history.ts';
import { useElementWidth, useLive } from '../lib/live.ts';
import { CHANNELS, type ChannelName } from '../lib/protocol.ts';
import { linear, nearestIndex, niceCeil, tickLabel } from '../lib/scale.ts';
import { HEALTH_LOOK } from '../lib/status.ts';
import { Meter, Panel, Segmented, StatusBadge, Waiting } from './ui.tsx';

const ROW_HEIGHT = 64;
const PAD_TOP = 8;
const GUTTER_RIGHT = 40;
/** A gap longer than this between samples is a disconnect: the line breaks. */
const MAX_GAP_S = 3;

interface Geometry {
  width: number;
  plotWidth: number;
  t0: number;
  t1: number;
  x: (t: number) => number;
}

function seriesPath(
  points: readonly EntropySample[],
  channel: ChannelName,
  g: Geometry,
  y: (v: number) => number,
) {
  let line = '';
  let area = '';
  let runStart = -1;
  const baseline = y(0);
  const close = (endX: number) => {
    if (runStart >= 0) area += `L${endX.toFixed(1)},${baseline}Z`;
  };
  points.forEach((p, i) => {
    const px = g.x(p.t).toFixed(1);
    const py = y(p.rate[channel]).toFixed(1);
    const prev = points[i - 1];
    if (!prev || p.t - prev.t > MAX_GAP_S) {
      if (prev) close(g.x(prev.t));
      line += `M${px},${py}`;
      area += `M${px},${baseline}L${px},${py}`;
      runStart = i;
    } else {
      line += `L${px},${py}`;
      area += `L${px},${py}`;
    }
  });
  const last = points.at(-1);
  if (last) close(g.x(last.t));
  return { line, area };
}

function nightBands(points: readonly EntropySample[]): [number, number][] {
  const bands: [number, number][] = [];
  let start: number | null = null;
  for (const p of points) {
    if (p.mode === 'night' && start === null) start = p.t;
    if (p.mode !== 'night' && start !== null) {
      bands.push([start, p.t]);
      start = null;
    }
  }
  const last = points.at(-1);
  if (start !== null && last) bands.push([start, last.t]);
  return bands;
}

function ChannelRow({
  channel,
  points,
  geometry,
  hover,
}: {
  channel: ChannelName;
  points: readonly EntropySample[];
  geometry: Geometry;
  hover: EntropySample | null;
}) {
  const max = niceCeil(Math.max(...points.map((p) => p.rate[channel])) * 1.12);
  const y = linear([0, max], [ROW_HEIGHT, PAD_TOP]);
  const { line, area } = seriesPath(points, channel, geometry, y);
  const color = CHANNEL_META[channel].color;
  const last = points.at(-1);
  const bands = nightBands(points);
  return (
    <svg className="spark-row__plot" width={geometry.width} height={ROW_HEIGHT + 1} aria-hidden>
      {bands.map(([a, b]) => (
        <rect
          key={a}
          x={geometry.x(a)}
          y={0}
          width={Math.max(1, geometry.x(b) - geometry.x(a))}
          height={ROW_HEIGHT}
          fill="var(--night-band)"
        />
      ))}
      <line x1={0} x2={geometry.plotWidth} y1={y(max)} y2={y(max)} className="chart-grid" />
      <line x1={0} x2={geometry.plotWidth} y1={y(0) + 0.5} y2={y(0) + 0.5} className="chart-axis" />
      <text x={geometry.width - 2} y={y(max) + 4} className="chart-tick" textAnchor="end">
        {tickLabel(max, max)}
      </text>
      <text x={geometry.width - 2} y={y(0) + 2} className="chart-tick" textAnchor="end">
        0
      </text>
      <path d={area} fill={color} fillOpacity={0.1} />
      <path
        d={line}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {hover ? (
        <>
          <line
            x1={geometry.x(hover.t)}
            x2={geometry.x(hover.t)}
            y1={0}
            y2={ROW_HEIGHT}
            className="chart-crosshair"
          />
          <circle
            cx={geometry.x(hover.t)}
            cy={y(hover.rate[channel])}
            r={4}
            fill={color}
            className="chart-dot"
          />
        </>
      ) : (
        last && (
          <circle
            cx={geometry.x(last.t)}
            cy={y(last.rate[channel])}
            r={4}
            fill={color}
            className="chart-dot"
          />
        )
      )}
    </svg>
  );
}

function TimeAxis({ geometry }: { geometry: Geometry }) {
  const span = geometry.t1 - geometry.t0;
  const step = geometry.plotWidth < 360 ? 120 : 60;
  const ticks: number[] = [];
  for (let ago = 0; ago <= span + 0.01; ago += step) ticks.push(ago);
  return (
    <svg className="spark-axis" width={geometry.width} height={18} aria-hidden>
      {ticks.map((ago) => {
        const x = geometry.x(geometry.t1 - ago);
        const anchor = ago === 0 ? 'end' : x < 12 ? 'start' : 'middle';
        return (
          <text key={ago} x={x} y={12} className="chart-tick" textAnchor={anchor}>
            {formatAgo(ago)}
          </text>
        );
      })}
    </svg>
  );
}

function EntropyTable({ points }: { points: readonly EntropySample[] }) {
  const last = points.at(-1);
  if (!last) return null;
  const rows: EntropySample[] = [];
  for (let ago = 0; ago <= 300; ago += 30) {
    const i = nearestIndex(points, last.t - ago, (p) => p.t);
    const p = points[i];
    if (p && Math.abs(p.t - (last.t - ago)) < 2) rows.push(p);
  }
  return (
    <div className="table-wrap">
      <table className="data-table">
        <caption className="sr-only">Credited entropy per channel, sampled every 30 seconds</caption>
        <thead>
          <tr>
            <th scope="col">Time</th>
            {CHANNELS.map((c) => (
              <th key={c} scope="col" className="num">
                {CHANNEL_META[c].label}
              </th>
            ))}
            <th scope="col">Mode</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr key={p.t}>
              <td>{formatAgo(Math.round(last.t - p.t))}</td>
              {CHANNELS.map((c) => (
                <td key={c} className="num">
                  {formatRate(p.rate[c])}
                </td>
              ))}
              <td>{p.mode}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Credited bits per second per channel as small multiples on a shared time
 * axis. Separate y-scales are deliberate: sensor noise earns ~10^4-10^5
 * bit/s and fish ~10^2, so one axis would flatten the fish to zero and a
 * dual axis would invent a correlation. Hovering (or arrow keys) moves one
 * crosshair through all rows; each row's readout shows the value under it.
 */
export function EntropyPanel({ history }: { history: EntropyHistory }) {
  const version = useSyncExternalStore(
    (notify) => history.subscribe(notify),
    () => history.version,
  );
  const latest = useLive('entropy');
  const [view, setView] = useState<'chart' | 'table'>('chart');
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [plotRef, plotWidth] = useElementWidth<HTMLDivElement>();
  // `version` is the change signal for the mutable history buffer.
  const points = useMemo(() => [...history.points], [history, version]);

  const geometry = useMemo<Geometry | null>(() => {
    const last = points.at(-1);
    if (!last || plotWidth <= 0) return null;
    const inner = Math.max(40, plotWidth - GUTTER_RIGHT);
    const t1 = last.t;
    const t0 = t1 - history.windowSeconds;
    return { width: plotWidth, plotWidth: inner, t0, t1, x: linear([t0, t1], [0, inner]) };
  }, [points, plotWidth, history.windowSeconds]);

  const hover = hoverIndex !== null ? (points[hoverIndex] ?? null) : null;
  const shown = hover ?? points.at(-1) ?? null;
  const night = points.some((p) => p.mode === 'night');

  const onPointer = (event: PointerEvent<HTMLDivElement>) => {
    if (!geometry) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.min(geometry.plotWidth, Math.max(0, event.clientX - rect.left));
    const t = geometry.t0 + (x / geometry.plotWidth) * (geometry.t1 - geometry.t0);
    setHoverIndex(nearestIndex(points, t, (p) => p.t));
  };

  const onKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (points.length === 0) return;
    const step = event.shiftKey ? 20 : 2;
    const current = hoverIndex ?? points.length - 1;
    if (event.key === 'ArrowLeft') setHoverIndex(Math.max(0, current - step));
    else if (event.key === 'ArrowRight') setHoverIndex(Math.min(points.length - 1, current + step));
    else if (event.key === 'Escape') setHoverIndex(null);
    else return;
    event.preventDefault();
  };

  const last = points.at(-1);
  return (
    <Panel
      className="area-entropy"
      title="Entropy credit"
      subtitle="Credited bits per second over the last 5 minutes. Each channel has its own scale; credit is min-entropy × 0.5 safety factor, and zero while a channel is unhealthy."
      actions={
        <Segmented
          label="Entropy view"
          value={view}
          onChange={setView}
          options={[
            { value: 'chart', label: 'Chart', icon: 'chart' },
            { value: 'table', label: 'Table', icon: 'table' },
          ]}
        />
      }
    >
      {!last || !latest ? (
        <Waiting>Waiting for entropy telemetry…</Waiting>
      ) : view === 'table' ? (
        <EntropyTable points={points} />
      ) : (
        <div
          className="spark"
          tabIndex={0}
          role="group"
          aria-label="Entropy per channel. Use the arrow keys to inspect values, Escape to reset."
          onKeyDown={onKey}
          onBlur={() => setHoverIndex(null)}
        >
          {CHANNELS.map((channel) => {
            const meta = CHANNEL_META[channel];
            const stats = latest.channels[channel];
            return (
              <div className="spark-row" key={channel}>
                <div className="spark-row__label">
                  <div className="spark-row__name">
                    <span className="line-key" style={{ background: meta.color }} aria-hidden />
                    {meta.label}
                  </div>
                  <div className="spark-row__value">{shown ? formatRate(shown.rate[channel]) : '-'}</div>
                  <div className="spark-row__meta">
                    {stats.h_per_sample !== null ? (
                      <span title="Rolling min-entropy estimate per sample (MCV)">
                        H {stats.h_per_sample.toFixed(2)} bit/sample
                      </span>
                    ) : (
                      <span>{meta.source}</span>
                    )}
                    <StatusBadge look={HEALTH_LOOK[stats.health]} size="sm" />
                  </div>
                </div>
                <div
                  className="spark-row__chart"
                  ref={channel === 'sensor' ? plotRef : undefined}
                  onPointerMove={onPointer}
                  onPointerLeave={() => setHoverIndex(null)}
                >
                  {geometry && (
                    <ChannelRow channel={channel} points={points} geometry={geometry} hover={hover} />
                  )}
                </div>
              </div>
            );
          })}
          <div className="spark-row spark-row--axis">
            <div className="spark-row__label spark-row__label--axis">
              <span className="spark__when" aria-live="polite">
                {hover ? `at ${formatAgo(Math.round(last.t - hover.t))}, ${hover.mode}` : 'latest values'}
              </span>
              {night && (
                <span className="legend-note">
                  <span className="legend-note__swatch" aria-hidden /> night (IR)
                </span>
              )}
            </div>
            <div className="spark-row__chart">{geometry && <TimeAxis geometry={geometry} />}</div>
          </div>
        </div>
      )}
      {latest && (
        <div className="pools">
          <div className="pools__head">
            <span className="eyebrow">Pool credit, undrawn</span>
            <span className="muted">
              resets on every draw · cap {POOL_CAPACITY_BITS.toLocaleString('en')} bit
            </span>
          </div>
          {(['sensor', 'fish'] as const).map((channel) => (
            <div className="pools__row" key={channel}>
              <span className="pools__name">{CHANNEL_META[channel].label}</span>
              <Meter
                value={latest.pool[channel]}
                max={POOL_CAPACITY_BITS}
                color={CHANNEL_META[channel].color}
                label={`${CHANNEL_META[channel].label} pool`}
              />
              <span className="pools__value">{formatBits(latest.pool[channel])}</span>
            </div>
          ))}
        </div>
      )}
    </Panel>
  );
}
