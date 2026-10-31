import { useEffect, useState } from 'react';

import { useConnectionStatus, useLive } from '../lib/live.ts';
import { applyTheme, readTheme, THEME_ORDER, type ThemePreference } from '../lib/theme.ts';
import type { ConnectionStatus } from '../lib/ws.ts';
import type { Tone } from '../lib/status.ts';
import { Icon, ToneDot } from './ui.tsx';

function connectionLook(status: ConnectionStatus): { tone: Tone; label: string } {
  switch (status.state) {
    case 'open':
      return { tone: 'good', label: 'Live' };
    case 'connecting':
      return { tone: 'warning', label: 'Connecting…' };
    case 'reconnecting':
      return {
        tone: 'serious',
        label: `Reconnecting in ${Math.max(1, Math.round((status.retryInMs ?? 0) / 1000))} s`,
      };
    case 'idle':
    case 'stopped':
      return { tone: 'neutral', label: 'Offline' };
  }
}

const THEME_LABEL: Record<ThemePreference, string> = {
  auto: 'Theme: follows system',
  light: 'Theme: light',
  dark: 'Theme: dark',
};

function ThemeToggle() {
  const [theme, setTheme] = useState<ThemePreference>(readTheme);
  useEffect(() => applyTheme(theme), [theme]);
  const next = THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length] ?? 'auto';
  const icon = theme === 'auto' ? 'auto' : theme === 'light' ? 'sun' : 'moon';
  return (
    <button
      type="button"
      className="icon-button icon-button--framed"
      onClick={() => setTheme(next)}
      aria-label={`${THEME_LABEL[theme]}. Switch to ${next}.`}
      title={THEME_LABEL[theme]}
    >
      <Icon name={icon} />
    </button>
  );
}

function BrandMark() {
  // A puffer fish reduced to geometry: body, spines, tail, eye.
  return (
    <svg className="brand__mark" viewBox="0 0 32 32" aria-hidden>
      <defs>
        <radialGradient id="brand-body" cx="40%" cy="35%" r="70%">
          <stop offset="0" stopColor="var(--accent)" stopOpacity="0.95" />
          <stop offset="1" stopColor="var(--accent)" stopOpacity="0.55" />
        </radialGradient>
      </defs>
      <g stroke="var(--accent)" strokeWidth="1.4" strokeLinecap="round">
        <path d="M13 4.2v2.4M18.6 4.6l-.8 2.2M8.2 7.4l1.6 1.8M23.2 8l-1.6 1.6M6 13H8.4M6.6 19.4l2-1.2M12.6 26.4l.6-2.2M18.4 26.6l-.4-2.2M22.8 23.8l-1.4-1.6" />
      </g>
      <circle cx="15.2" cy="15.6" r="8.4" fill="url(#brand-body)" />
      <path d="M23.2 15.6 29 11.4v8.4Z" fill="var(--accent)" opacity="0.7" />
      <circle cx="11.8" cy="13.4" r="1.6" fill="var(--surface)" />
    </svg>
  );
}

export function Header() {
  const status = useConnectionStatus();
  const hello = useLive('hello');
  const conn = connectionLook(status);
  return (
    <header className="topbar">
      <div className="brand">
        <BrandMark />
        <div>
          <div className="brand__name">puffer-cipher</div>
          <div className="brand__tag">Entropy from living fish</div>
        </div>
      </div>
      <div className="topbar__meta">
        {hello && (
          <span className="topbar__source" title="Frame source reported by the server">
            <span className="topbar__source-name">{hello.source}</span>
            <span className="topbar__source-spec">
              {hello.width}×{hello.height} · {hello.fps} fps
            </span>
          </span>
        )}
        <span
          className={`conn conn--${conn.tone}`}
          role="status"
          title={status.rejected > 0 ? `${status.rejected} malformed messages dropped` : undefined}
        >
          <ToneDot tone={conn.tone} pulse={status.state === 'open'} />
          {conn.label}
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
