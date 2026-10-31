import { useMemo } from 'react';

import { api } from '../lib/api.ts';
import { formatBits, formatCount, formatDuration } from '../lib/format.ts';
import { useLive, usePolling } from '../lib/live.ts';
import type { FishCard, LeaderboardEntry } from '../lib/protocol.ts';
import { Icon, Meter, Panel, StatusBadge, Waiting } from './ui.tsx';

/** Fish cards come over the socket every 5 s; REST fills the gap until the first one. */
export function useFishCards(): FishCard[] | undefined {
  const live = useLive('fish');
  const initial = usePolling(api.fish, 0);
  return live?.fish ?? initial.data;
}

function FishCardView({ fish }: { fish: FishCard }) {
  const s = fish.stats;
  return (
    <article className={`fish-card${fish.present ? '' : ' fish-card--away'}`}>
      <header className="fish-card__head">
        <span className="swatch swatch--lg" style={{ background: fish.color }} aria-hidden />
        <div className="fish-card__titles">
          <h3 className="fish-card__name">{fish.name}</h3>
          <div className="fish-card__ids mono">
            {fish.fish_id}
            {fish.track_id !== null && <> · track #{fish.track_id}</>}
          </div>
        </div>
        <StatusBadge
          size="sm"
          look={
            fish.present
              ? { tone: 'good', icon: 'check', label: 'In view' }
              : { tone: 'neutral', icon: 'minus', label: 'Hidden' }
          }
        />
      </header>
      <dl className="fish-card__stats">
        <div>
          <dt>Speed</dt>
          <dd>
            {Math.round(s.speed_avg)} <span className="unit">px/s</span>
          </dd>
        </div>
        <div>
          <dt>Top speed</dt>
          <dd>
            {Math.round(s.speed_max)} <span className="unit">px/s</span>
          </dd>
        </div>
        <div>
          <dt>Active</dt>
          <dd>{Math.round(s.activity * 100)}%</dd>
        </div>
        <div>
          <dt>Bits today</dt>
          <dd>{formatBits(s.bits_today)}</dd>
        </div>
        <div>
          <dt>Lifetime</dt>
          <dd>{formatBits(s.bits_total)}</dd>
        </div>
        <div>
          <dt>Distance</dt>
          <dd>
            {formatCount(s.distance_px)} <span className="unit">px</span>
          </dd>
        </div>
      </dl>
      <div className="fish-card__chaos">
        <span className="eyebrow">Chaos</span>
        <Meter value={s.chaos_score} max={1} color="var(--ink-2)" label={`${fish.name} chaos score`} />
        <span className="fish-card__chaos-value">{s.chaos_score.toFixed(2)}</span>
      </div>
      <div className="fish-card__seen muted">seen for {formatDuration(s.seen_s)}</div>
    </article>
  );
}

export function FishPanel({ fish }: { fish: FishCard[] | undefined }) {
  const present = fish?.filter((f) => f.present).length ?? 0;
  return (
    <Panel
      className="area-fish"
      title="The fish"
      subtitle="Identities come from colour re-identification, so a fish keeps its name when the tracker loses and re-acquires it."
      actions={
        fish && (
          <span className="chip chip--quiet">
            {present} of {fish.length} in view
          </span>
        )
      }
    >
      {!fish ? (
        <Waiting>Waiting for fish…</Waiting>
      ) : (
        <div className="fish-grid">
          {fish.map((f) => (
            <FishCardView key={f.fish_id} fish={f} />
          ))}
        </div>
      )}
    </Panel>
  );
}

/**
 * Ranked bars, emphasis form: the leader in the fish-channel hue, everyone
 * else in the de-emphasis grey. Every value is printed at its bar's tip, so
 * nothing depends on hovering.
 */
export function LeaderboardPanel({ fish }: { fish: FishCard[] | undefined }) {
  const { data, error } = usePolling(() => api.leaderboard(7), 60_000);
  const colors = useMemo(() => new Map(fish?.map((f) => [f.fish_id, f.color])), [fish]);
  const max = Math.max(1e-9, ...(data ?? []).map((e) => e.chaos_score));
  return (
    <Panel
      className="area-leaderboard"
      title="Most chaotic fish"
      subtitle="Mean chaos score over 7 days: the turns and accelerations the tracker did not see coming."
    >
      {!data ? (
        <Waiting>{error ? `Leaderboard unavailable: ${error.message}` : 'Loading leaderboard…'}</Waiting>
      ) : data.length === 0 ? (
        <Waiting>No fish ranked yet.</Waiting>
      ) : (
        <ol className="ranking">
          {data.map((entry: LeaderboardEntry) => {
            const lead = entry.rank === 1;
            return (
              <li key={entry.fish_id} className={`ranking__row${lead ? ' ranking__row--lead' : ''}`}>
                <span className="ranking__rank">
                  {lead ? <Icon name="trophy" size={15} label="Rank 1" /> : entry.rank}
                </span>
                <span className="ranking__who">
                  <span
                    className="swatch"
                    style={{ background: colors.get(entry.fish_id) ?? 'var(--neutral)' }}
                    aria-hidden
                  />
                  <span className="ranking__name">{entry.name}</span>
                  <span className="ranking__bits">{formatBits(entry.bits)}</span>
                </span>
                <span className="ranking__bar-wrap">
                  <span
                    className="ranking__bar"
                    style={{ width: `calc((100% - 3.2em) * ${entry.chaos_score / max})` }}
                    title={`${entry.name}: chaos ${entry.chaos_score.toFixed(3)}, ${formatBits(entry.bits)} credited`}
                  />
                  <span className="ranking__value">{entry.chaos_score.toFixed(2)}</span>
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
