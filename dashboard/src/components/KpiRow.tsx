import { type CSSProperties, type ReactNode, useMemo, useSyncExternalStore } from 'react';

import { api } from '../lib/api.ts';
import { formatDuration, formatRate } from '../lib/format.ts';
import type { EntropyHistory } from '../lib/history.ts';
import { useLive, useNow, usePolling } from '../lib/live.ts';
import type { FishCard } from '../lib/protocol.ts';
import { linear } from '../lib/scale.ts';
import { verifyPulse } from '../lib/beacon.ts';
import { CHECK_LOOK, LIVENESS_LOOK } from '../lib/status.ts';
import { StatusBadge } from './ui.tsx';

const SPARK_W = 112;
const SPARK_H = 30;

/** 5-minute trend in the de-emphasis ink; the current value is the accent dot. */
function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const max = Math.max(...values) * 1.05;
  const min = Math.min(...values) * 0.95;
  const x = linear([0, values.length - 1], [2, SPARK_W - 5]);
  const y = linear([min, max], [SPARK_H - 3, 3]);
  const d = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('');
  const last = values.at(-1) ?? 0;
  return (
    <svg className="kpi__spark" width={SPARK_W} height={SPARK_H} aria-hidden>
      <path d={d} fill="none" stroke="var(--ink-3)" strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={x(values.length - 1)} cy={y(last)} r={3.5} fill="var(--accent)" className="chart-dot" />
    </svg>
  );
}

function Kpi({
  label,
  value,
  detail,
  aside,
}: {
  label: string;
  value: ReactNode;
  detail: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <div className="kpi">
      <div className="kpi__text">
        <div className="kpi__label">{label}</div>
        <div className="kpi__value">{value}</div>
        <div className="kpi__detail">{detail}</div>
      </div>
      {aside && <div className="kpi__aside">{aside}</div>}
    </div>
  );
}

export function KpiRow({ history, fish }: { history: EntropyHistory; fish: FishCard[] | undefined }) {
  const version = useSyncExternalStore(
    (notify) => history.subscribe(notify),
    () => history.version,
  );
  const entropy = useLive('entropy');
  const led = useLive('led');
  const livePulse = useLive('beacon');
  const firstPulse = usePolling(api.beaconLatest, 0);
  const now = useNow(1000);

  const trend = useMemo(() => {
    const points = history.points;
    const step = Math.max(1, Math.floor(points.length / 48));
    const out: number[] = [];
    for (let i = 0; i < points.length; i += step) {
      const p = points[i];
      if (p) out.push(p.rate.sensor + p.rate.fish);
    }
    return out;
    // `version` is the change signal for the mutable history buffer.
  }, [history, version]);

  const camera = entropy ? entropy.channels.sensor.bits_per_s + entropy.channels.fish.bits_per_s : null;
  const present = fish?.filter((f) => f.present) ?? [];
  const hidden = fish?.filter((f) => !f.present) ?? [];
  const hello = useLive('hello');
  const pulse = livePulse ?? firstPulse.data;
  // Standalone check of the newest pulse against the pinned key (the beacon
  // panel below also checks the chain links).
  const signed = useMemo(() => {
    if (!pulse || !hello?.beacon_public_key) return null;
    const { checks } = verifyPulse(pulse, { publicKey: hello.beacon_public_key });
    return [checks.public_key, checks.signature, checks.output, checks.summary_hash].every(
      (c) => c === 'pass',
    );
  }, [pulse, hello?.beacon_public_key]);
  const nextIn = pulse ? (pulse.timestamp_ms + pulse.period_ms - now) / 1000 : null;

  return (
    <div className="kpis area-kpis">
      <Kpi
        label="Camera entropy"
        value={camera !== null ? formatRate(camera) : '-'}
        detail="sensor + fish, credited"
        aside={<Sparkline values={trend} />}
      />
      <Kpi
        label="Fish in view"
        value={
          fish ? (
            <>
              {present.length}
              <span className="kpi__of"> / {fish.length}</span>
            </>
          ) : (
            '-'
          )
        }
        detail={hidden.length > 0 ? `${hidden.map((f) => f.name).join(', ')} hiding` : 'all fish tracked'}
        aside={
          fish && (
            <span className="kpi__school" aria-hidden>
              {fish.map((f) => (
                <span
                  key={f.fish_id}
                  className={`kpi__fish${f.present ? '' : ' kpi__fish--away'}`}
                  style={{ '--fish': f.color } as CSSProperties}
                />
              ))}
            </span>
          )
        }
      />
      <Kpi
        label="Liveness confidence"
        value={
          led ? (
            <>
              {led.confidence_bits.toFixed(1)}
              <span className="kpi__of"> bits</span>
            </>
          ) : (
            '-'
          )
        }
        detail={led ? `camera lag ${Math.round(led.lag_s * 1000)} ms` : 'LED challenge'}
        aside={led && <StatusBadge look={LIVENESS_LOOK[led.state]} size="sm" />}
      />
      <Kpi
        label="Beacon height"
        value={pulse ? `#${pulse.index.toLocaleString('en')}` : '-'}
        detail={
          nextIn === null
            ? 'signed public randomness'
            : nextIn > 0
              ? `next pulse in ${formatDuration(nextIn)}`
              : 'next pulse due'
        }
        aside={
          signed !== null && (
            <StatusBadge
              look={signed ? CHECK_LOOK.pass : CHECK_LOOK.fail}
              size="sm"
              label={signed ? 'Signed' : 'Bad signature'}
            />
          )
        }
      />
    </div>
  );
}
