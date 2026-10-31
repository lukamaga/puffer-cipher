import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api } from '../lib/api.ts';
import { type PulseCheck, type PulseVerification, verifyPulse } from '../lib/beacon.ts';
import { formatBits, formatClock, formatDuration, groupHex, shortHex } from '../lib/format.ts';
import { useLive, useLiveClient, useNow, usePolling } from '../lib/live.ts';
import { type BeaconPulse, parsePulseSummary } from '../lib/protocol.ts';
import { CHECK_LOOK } from '../lib/status.ts';
import { CopyButton, Icon, Panel, StatusBadge, Waiting } from './ui.tsx';

const PAGE = 5;

const CHECK_LABEL: Record<PulseCheck, string> = {
  public_key: 'Key',
  summary_hash: 'Summary',
  signature: 'Signature',
  output: 'Output',
  genesis: 'Genesis',
  index: 'Index',
  timestamp: 'Time',
  link: 'Chain link',
  precommitment: 'Precommit',
};

const CHECK_HELP: Record<PulseCheck, string> = {
  public_key: 'The pulse names the pinned beacon key',
  summary_hash: 'BLAKE2b-256 of the summary string matches summary_hash',
  signature: 'Ed25519 signature over the 260-byte message verifies under the pinned key',
  output: 'output = BLAKE2b-512(message || signature)',
  genesis: 'Pulse 0 links to 64 zero bytes',
  index: 'Index is exactly one more than the previous pulse',
  timestamp: 'Timestamp is later than the previous pulse',
  link: 'prev_output equals the previous pulse output',
  precommitment: 'BLAKE2b-512(local_random) equals the previous pulse precommitment',
};

/** Checks always shown; index/time are folded into "Chain link" unless they fail. */
const PRIMARY: PulseCheck[] = ['signature', 'output', 'summary_hash', 'link', 'precommitment'];

function CheckChips({ result }: { result: PulseVerification }) {
  const extra = (['public_key', 'genesis', 'index', 'timestamp'] as const).filter(
    (c) => result.checks[c] === 'fail' || (c === 'genesis' && result.checks.genesis === 'pass'),
  );
  return (
    <ul className="checks checks--pulse">
      {[...PRIMARY, ...extra].map((check) => {
        const outcome = result.checks[check];
        const look = CHECK_LOOK[outcome];
        const help =
          outcome === 'skip' ? `${CHECK_HELP[check]} - needs the previous pulse` : CHECK_HELP[check];
        return (
          <li key={check} className={`check check--${look.tone}`} title={`${help}: ${look.label}`}>
            <Icon name={look.icon} size={12} />
            {CHECK_LABEL[check]}
          </li>
        );
      })}
    </ul>
  );
}

function PulseRow({ pulse, result, now }: { pulse: BeaconPulse; result: PulseVerification; now: number }) {
  const summary = parsePulseSummary(pulse.summary);
  const age = Math.max(0, (now - pulse.timestamp_ms) / 1000);
  return (
    <li className={`pulse${result.valid ? '' : ' pulse--invalid'}`}>
      <div className="pulse__main">
        <div className="pulse__id">
          <span className="pulse__index">#{pulse.index.toLocaleString('en')}</span>
          <span className="pulse__time">
            {formatClock(pulse.timestamp_ms)} <span className="muted">· {formatDuration(age)} ago</span>
          </span>
        </div>
        <div className="pulse__output mono" title={pulse.output}>
          {shortHex(pulse.output, 16, 8)}
        </div>
        <StatusBadge
          size="sm"
          look={result.valid ? CHECK_LOOK.pass : CHECK_LOOK.fail}
          label={result.valid ? 'Verified' : (result.error ?? 'invalid')}
        />
      </div>
      <CheckChips result={result} />
      <details className="pulse__raw">
        <summary>
          <span className="pulse__summary muted">
            {summary ? (
              <>
                {summary.fish_count} fish · {formatBits(summary.fish_bits)} fish +{' '}
                {formatBits(summary.sensor_bits)} sensor
                {summary.top_fish[0] && <> · top: {summary.top_fish[0].name}</>}
              </>
            ) : (
              'summary not in the documented format'
            )}
          </span>
          <span className="pulse__raw-toggle">
            JSON <Icon name="chevron" size={14} />
          </span>
        </summary>
        <pre className="mono">{JSON.stringify(pulse, null, 2)}</pre>
      </details>
    </li>
  );
}

/**
 * Beacon feed. The server's key is pinned once (from `hello`, falling back to
 * REST); every pulse is then checked in the browser against that key and
 * against its predecessor. The server is never asked whether a pulse is
 * valid - that is the point.
 */
export function BeaconPanel() {
  const client = useLiveClient();
  const hello = useLive('hello');
  const restKey = usePolling(api.beaconPublicKey, 0);
  const now = useNow(1000);
  const [pulses, setPulses] = useState<Map<number, BeaconPulse>>(new Map());
  const [shown, setShown] = useState(PAGE);
  const [error, setError] = useState<string | null>(null);
  const [empty, setEmpty] = useState(false);
  const [fork, setFork] = useState<{ index: number; at: number } | null>(null);

  const pinnedKey = hello?.beacon_public_key ?? restKey.data?.public_key ?? null;
  const keyMismatch =
    hello?.beacon_public_key &&
    restKey.data &&
    hello.beacon_public_key.toLowerCase() !== restKey.data.public_key.toLowerCase();

  const merge = useCallback((incoming: BeaconPulse[]) => {
    if (incoming.length === 0) return;
    setPulses((prev) => {
      const next = new Map(prev);
      for (const p of incoming) next.set(p.index, p);
      return next;
    });
  }, []);

  const load = useCallback(
    async (start: number, count: number) => {
      try {
        merge(await api.beaconPulses(Math.max(0, start), count));
        setError(null);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    },
    [merge],
  );

  // Mirror of `pulses` for socket callbacks and resync, which run outside render.
  const held = useRef(pulses);
  useEffect(() => {
    held.current = pulses;
  }, [pulses]);

  // (Re)sync on mount and on every handshake: fetch the latest pulse, the
  // ones before it, and one extra so the oldest visible pulse can still be
  // chain-checked. A server whose chain is not a continuation of what this
  // page already saw (head moved backwards, or a different pulse at an index
  // we hold) is equivocating or was reset. That is surfaced, never merged:
  // mixing the two histories would fabricate a chain nobody published.
  const resync = useCallback(async () => {
    try {
      const latest = await api.beaconLatest();
      const page = [...(await api.beaconPulses(Math.max(0, latest.index - PAGE), PAGE + 1)), latest];
      const prev = held.current;
      const conflict = page.find((p) => {
        const seen = prev.get(p.index);
        return seen !== undefined && seen.output !== p.output;
      });
      const headMovedBack = latest.index < Math.max(-1, ...prev.keys());
      if (conflict || headMovedBack) setFork({ index: conflict?.index ?? latest.index, at: Date.now() });
      const next = new Map(conflict || headMovedBack ? [] : prev);
      for (const p of page) next.set(p.index, p);
      setPulses(next);
      setEmpty(false);
      setError(null);
    } catch (e) {
      if (e instanceof Error && 'status' in e && e.status === 404) setEmpty(true);
      else setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void resync();
    return client.on('hello', () => void resync());
  }, [client, resync]);

  // Live pulses; a gap (missed while disconnected) is back-filled over REST.
  useEffect(
    () =>
      client.on('beacon', (pulse) => {
        setEmpty(false);
        const known = Math.max(-1, ...held.current.keys());
        if (known >= 0 && pulse.index > known + 1) void load(known + 1, pulse.index - known - 1);
        merge([pulse]);
      }),
    [client, load, merge],
  );

  const ordered = useMemo(() => [...pulses.values()].sort((a, b) => b.index - a.index), [pulses]);
  const results = useMemo(() => {
    const out = new Map<number, PulseVerification>();
    for (const p of ordered) {
      out.set(p.index, verifyPulse(p, { publicKey: pinnedKey, prev: pulses.get(p.index - 1) ?? null }));
    }
    return out;
  }, [ordered, pulses, pinnedKey]);

  const visible = ordered.slice(0, shown);
  const latest = ordered[0];
  const failures = visible.filter((p) => !results.get(p.index)?.valid).length;
  const nextIn = latest ? (latest.timestamp_ms + latest.period_ms - now) / 1000 : null;
  const oldest = visible.at(-1);

  return (
    <Panel
      className="area-beacon"
      title="Randomness beacon"
      subtitle="Signed, hash-chained public randomness, re-verified in your browser with Ed25519 and BLAKE2b. Beacon values are public: never use them as keys."
      actions={
        latest && (
          <StatusBadge
            look={failures === 0 ? CHECK_LOOK.pass : CHECK_LOOK.fail}
            label={failures === 0 ? 'Chain verified' : `${failures} invalid`}
          />
        )
      }
    >
      <div className="beacon-key">
        <div className="beacon-key__head">
          <span className="eyebrow">
            <Icon name="key" size={13} /> Pinned Ed25519 public key
          </span>
          {pinnedKey && <CopyButton text={pinnedKey} label="Copy public key" />}
        </div>
        <div className="mono beacon-key__value">{pinnedKey ? groupHex(pinnedKey, 8) : 'not available'}</div>
        <div className="muted">
          {hello?.beacon_public_key ? 'from the live feed handshake' : 'from /api/beacon/public-key'}
          {keyMismatch && (
            <strong className="text-critical"> · REST and live feed disagree on the key</strong>
          )}
        </div>
      </div>
      {latest && (
        <div className="beacon-meta">
          <span>
            Height <strong>#{latest.index.toLocaleString('en')}</strong>
          </span>
          <span>period {formatDuration(latest.period_ms / 1000)}</span>
          {nextIn !== null && (
            <span>{nextIn > 0 ? `next pulse in ${formatDuration(nextIn)}` : 'next pulse due'}</span>
          )}
        </div>
      )}
      {fork && (
        <div className="outcome outcome--critical" role="alert">
          <Icon name="alert" size={16} />
          <div>
            <strong>The server&apos;s chain diverged at pulse #{fork.index.toLocaleString('en')}</strong>
            <p className="outcome__text">
              At {formatClock(fork.at)} the beacon served a history that does not extend the pulses this page
              saw earlier: a reset, or equivocation. The earlier pulses were set aside rather than merged.
            </p>
          </div>
        </div>
      )}
      {error && (
        <p className="inline-error" role="alert">
          <Icon name="x" size={14} /> {error}
        </p>
      )}
      {empty && !latest ? (
        <Waiting>No pulses yet. The first one is published one period after start-up.</Waiting>
      ) : !latest ? (
        <Waiting>Loading pulses…</Waiting>
      ) : (
        <>
          <ol className="pulses">
            {visible.map((p) => {
              const result = results.get(p.index);
              return result ? <PulseRow key={p.index} pulse={p} result={result} now={now} /> : null;
            })}
          </ol>
          {oldest && oldest.index > 0 && (
            <button
              type="button"
              className="button button--ghost"
              onClick={() => {
                setShown((n) => n + PAGE);
                void load(oldest.index - PAGE - 1, PAGE + 1);
              }}
            >
              Load older pulses
            </button>
          )}
        </>
      )}
    </Panel>
  );
}
