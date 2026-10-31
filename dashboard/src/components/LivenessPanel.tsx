import { useLive } from '../lib/live.ts';
import { LIVENESS_LOOK } from '../lib/status.ts';
import { Meter, Panel, StatusBadge, Waiting } from './ui.tsx';

/** DESIGN 5: the observed window must match the challenge at >= 90%. */
const MATCH_THRESHOLD = 0.9;

function Strip({ bits, reference, label }: { bits: string; reference?: string; label: string }) {
  const mismatches = reference ? [...bits].filter((b, i) => b !== reference[i]).length : 0;
  return (
    <div className="strip">
      <div className="strip__label">
        <span>{label}</span>
        {reference && (
          <span className={mismatches > 0 ? 'strip__miss' : 'muted'}>
            {mismatches === 0 ? 'no mismatches' : `${mismatches} mismatch${mismatches > 1 ? 'es' : ''}`}
          </span>
        )}
      </div>
      <div
        className="strip__cells"
        style={{ gridTemplateColumns: `repeat(${Math.max(1, bits.length)}, 1fr)` }}
        aria-label={`${label}: ${bits}`}
        role="img"
      >
        {[...bits].map((b, i) => {
          const miss = reference !== undefined && reference[i] !== b;
          return (
            <span key={i} className={`cell${b === '1' ? ' cell--on' : ''}${miss ? ' cell--miss' : ''}`} />
          );
        })}
      </div>
    </div>
  );
}

/**
 * Anti-replay liveness (DESIGN 5). The server blinks an LED inside the tank
 * with fresh random bits; only a live camera can show the pattern that was
 * commanded moments ago. `confidence_bits` is -log2 of the chance that a
 * replayed or fake video matches this well by luck.
 */
export function LivenessPanel() {
  const led = useLive('led');
  return (
    <Panel
      className="area-liveness"
      title="LED liveness"
      subtitle="A fresh random LED pattern in the tank proves the video is live, not a replay."
      actions={led && <StatusBadge look={LIVENESS_LOOK[led.state]} />}
    >
      {!led ? (
        <Waiting>Waiting for the LED verifier…</Waiting>
      ) : led.state === 'disabled' ? (
        <p className="muted">
          No LED is configured for this source, so liveness is not checked (expected for recorded video).
        </p>
      ) : (
        <>
          <div className="liveness__hero">
            <div>
              <div className="figure">
                {led.confidence_bits.toFixed(1)}
                <span className="figure__unit">bits</span>
              </div>
              <div className="muted">
                confidence · chance match ≈ 2<sup>−{Math.round(led.confidence_bits)}</sup>
              </div>
            </div>
          </div>
          <Strip label="Commanded" bits={led.expected} />
          <Strip label="Observed" bits={led.observed} reference={led.expected} />
          <dl className="kv-grid kv-grid--3">
            <div>
              <dt>Match</dt>
              <dd>{(led.match_ratio * 100).toFixed(1)}%</dd>
              <Meter
                value={led.match_ratio}
                max={1}
                color={led.match_ratio >= MATCH_THRESHOLD ? 'var(--good)' : 'var(--critical)'}
                label="Match ratio"
                marker={{ at: MATCH_THRESHOLD, label: 'required: 90%' }}
              />
            </div>
            <div>
              <dt>Camera lag</dt>
              <dd>{Math.round(led.lag_s * 1000)} ms</dd>
            </div>
            <div>
              <dt>Window</dt>
              <dd>{(led.window_end - led.window_start).toFixed(1)} s</dd>
            </div>
          </dl>
        </>
      )}
    </Panel>
  );
}
