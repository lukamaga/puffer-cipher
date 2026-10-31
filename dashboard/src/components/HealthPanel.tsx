import { CHANNEL_META } from '../lib/channels.ts';
import { useLive } from '../lib/live.ts';
import type { ChannelHealth, HealthState } from '../lib/protocol.ts';
import { checkLabel, HEALTH_LOOK } from '../lib/status.ts';
import { Icon, Panel, StatusBadge, Waiting } from './ui.tsx';

const ORDER: Record<HealthState, number> = { failed: 0, degraded: 1, startup: 2, ok: 3 };

function channelLabel(name: string): string {
  return name in CHANNEL_META ? CHANNEL_META[name as keyof typeof CHANNEL_META].label : name;
}

function ChannelBlock({ name, health }: { name: string; health: ChannelHealth }) {
  // Problems first: a reader scanning for "what is wrong" should not have to search.
  const checks = Object.entries(health.checks).sort(([, a], [, b]) => ORDER[a] - ORDER[b]);
  return (
    <div className="health-channel">
      <div className="health-channel__head">
        <span className="health-channel__name">{channelLabel(name)}</span>
        <StatusBadge look={HEALTH_LOOK[health.state]} size="sm" />
      </div>
      <ul className="checks" aria-label={`${channelLabel(name)} checks`}>
        {checks.map(([check, state]) => {
          const look = HEALTH_LOOK[state];
          return (
            <li
              key={check}
              className={`check check--${look.tone}`}
              title={`${checkLabel(check)}: ${look.label}`}
            >
              <Icon name={look.icon} size={12} />
              {checkLabel(check)}
            </li>
          );
        })}
      </ul>
      {health.reasons.length > 0 && (
        <ul className="reasons">
          {health.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * SP 800-90B continuous health tests plus domain checks, per channel
 * (DESIGN 4.2). A failed channel is neither absorbed nor credited. The
 * system channel has no camera tests: it is os.urandom, mixed into every
 * draw unconditionally.
 */
export function HealthPanel() {
  const health = useLive('health');
  return (
    <Panel
      className="area-health"
      title="Source health"
      subtitle="Continuous SP 800-90B tests and domain checks. A failed channel earns no credit."
      actions={
        health && (
          <StatusBadge
            look={HEALTH_LOOK[health.overall]}
            label={`Overall: ${HEALTH_LOOK[health.overall].label}`}
          />
        )
      }
    >
      {!health ? (
        <Waiting>Waiting for health reports…</Waiting>
      ) : (
        <div className="health">
          {Object.entries(health.channels).map(([name, channel]) => (
            <ChannelBlock key={name} name={name} health={channel} />
          ))}
          {!('system' in health.channels) && (
            <p className="health__note muted">
              <strong>System</strong> has no camera tests: os.urandom(32) keys every draw, so output is never
              weaker than the OS.
            </p>
          )}
        </div>
      )}
    </Panel>
  );
}
