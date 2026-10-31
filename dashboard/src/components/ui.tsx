/**
 * Small presentational primitives shared by every panel. No UI framework:
 * plain elements with BEM-ish class names styled in `styles/app.css`.
 */

import { type CSSProperties, type ReactNode, useId, useState } from 'react';

import type { IconName, StatusLook, Tone } from '../lib/status.ts';

export type GlyphName =
  | IconName
  | 'sun'
  | 'moon'
  | 'auto'
  | 'copy'
  | 'lock'
  | 'unlock'
  | 'shield'
  | 'key'
  | 'trophy'
  | 'chevron'
  | 'zap'
  | 'chart'
  | 'table';

const GLYPHS: Record<GlyphName, ReactNode> = {
  check: <path d="M3.5 8.4 6.6 11.4 12.6 4.6" />,
  x: <path d="M4.8 4.8l6.4 6.4M11.2 4.8l-6.4 6.4" />,
  alert: (
    <>
      <path d="M8 2.4 14.2 13.2H1.8Z" />
      <path d="M8 6.6v3M8 11.4v.2" />
    </>
  ),
  clock: (
    <>
      <circle cx="8" cy="8" r="5.8" />
      <path d="M8 4.8V8l2.2 1.5" />
    </>
  ),
  minus: <path d="M4.5 8h7" />,
  sun: (
    <>
      <circle cx="8" cy="8" r="2.8" />
      <path d="M8 1.6v1.6M8 12.8v1.6M1.6 8h1.6M12.8 8h1.6M3.5 3.5l1.1 1.1M11.4 11.4l1.1 1.1M3.5 12.5l1.1-1.1M11.4 4.6l1.1-1.1" />
    </>
  ),
  moon: <path d="M13.2 9.8A5.6 5.6 0 0 1 6.2 2.8a5.6 5.6 0 1 0 7 7Z" />,
  auto: (
    <>
      <circle cx="8" cy="8" r="5.8" />
      <path d="M8 2.2v11.6A5.8 5.8 0 0 0 8 2.2Z" fill="currentColor" stroke="none" />
    </>
  ),
  copy: (
    <>
      <rect x="5.5" y="5.5" width="8" height="8" rx="1.6" />
      <path d="M10.5 3.6V3.4A1.4 1.4 0 0 0 9.1 2H3.4A1.4 1.4 0 0 0 2 3.4v5.7a1.4 1.4 0 0 0 1.4 1.4h.2" />
    </>
  ),
  lock: (
    <>
      <rect x="3" y="7" width="10" height="7" rx="1.6" />
      <path d="M5.2 7V5.1a2.8 2.8 0 0 1 5.6 0V7" />
    </>
  ),
  unlock: (
    <>
      <rect x="3" y="7" width="10" height="7" rx="1.6" />
      <path d="M5.2 7V5.1a2.8 2.8 0 0 1 5.4-1" />
    </>
  ),
  shield: <path d="M8 1.8 13.2 3.8v4c0 3-2.2 5.3-5.2 6.4-3-1.1-5.2-3.4-5.2-6.4v-4Z" />,
  key: (
    <>
      <circle cx="5.4" cy="10.6" r="2.8" />
      <path d="M7.4 8.6 13.4 2.6M11.2 4.8l1.6 1.6M9.6 6.4l1.2 1.2" />
    </>
  ),
  trophy: (
    <>
      <path d="M5 2.5h6v3.2a3 3 0 0 1-6 0Z" />
      <path d="M5 3.6H2.8a2 2 0 0 0 2.4 3M11 3.6h2.2a2 2 0 0 1-2.4 3M8 8.7v2.6M5.6 13.5h4.8M6.4 11.3h3.2" />
    </>
  ),
  chevron: <path d="M5.5 6.5 8 9l2.5-2.5" />,
  zap: <path d="M9 1.8 3.6 9h4l-.6 5.2L12.4 7h-4Z" />,
  chart: <path d="M2.5 12.5 6 8.5l2.6 2.2 4.9-6.2" />,
  table: (
    <>
      <rect x="2.4" y="3" width="11.2" height="10" rx="1.4" />
      <path d="M2.4 6.6h11.2M2.4 9.8h11.2M6.6 3v10" />
    </>
  ),
};

export function Icon({ name, size = 16, label }: { name: GlyphName; size?: number; label?: string }) {
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      role={label ? 'img' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      {GLYPHS[name]}
    </svg>
  );
}

/** Status pill: tone colour on the icon, label in ink - never colour alone. */
export function StatusBadge({
  look,
  label,
  size = 'md',
  title,
}: {
  look: StatusLook;
  label?: ReactNode;
  size?: 'sm' | 'md';
  title?: string;
}) {
  return (
    <span className={`badge badge--${look.tone} badge--${size}`} title={title}>
      <Icon name={look.icon} size={size === 'sm' ? 12 : 14} />
      <span>{label ?? look.label}</span>
    </span>
  );
}

export function ToneDot({ tone, pulse = false }: { tone: Tone; pulse?: boolean }) {
  return <span className={`dot dot--${tone}${pulse ? ' dot--pulse' : ''}`} aria-hidden />;
}

export function Panel({
  title,
  subtitle,
  actions,
  className = '',
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  const id = useId();
  return (
    <section className={`panel ${className}`} aria-labelledby={id}>
      <header className="panel__head">
        <div className="panel__titles">
          <h2 id={id} className="panel__title">
            {title}
          </h2>
          {subtitle && <p className="panel__subtitle">{subtitle}</p>}
        </div>
        {actions && <div className="panel__actions">{actions}</div>}
      </header>
      <div className="panel__body">{children}</div>
    </section>
  );
}

/** Ratio against a limit. The track is a lighter step of the fill's own hue. */
export function Meter({
  value,
  max,
  color,
  label,
  marker,
}: {
  value: number;
  max: number;
  color: string;
  label: string;
  marker?: { at: number; label: string };
}) {
  const fraction = max > 0 ? Math.min(1, Math.max(0, value / max)) : 0;
  return (
    <div
      className="meter"
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={Math.round(value)}
      style={{ '--meter-color': color } as CSSProperties}
    >
      <div className="meter__fill" style={{ width: `${fraction * 100}%` }} />
      {marker && (
        <div className="meter__marker" style={{ left: `${(marker.at / max) * 100}%` }} title={marker.label} />
      )}
    </div>
  );
}

/** Two or three mutually exclusive options (chart/table, presets). */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string; icon?: GlyphName }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          className="segmented__option"
          onClick={() => onChange(o.value)}
        >
          {o.icon && <Icon name={o.icon} size={14} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="icon-button"
      aria-label={copied ? 'Copied' : label}
      title={copied ? 'Copied' : label}
      onClick={() => {
        navigator.clipboard?.writeText(text).then(
          () => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          },
          () => undefined,
        );
      }}
    >
      <Icon name={copied ? 'check' : 'copy'} size={14} />
    </button>
  );
}

/** Quiet placeholder while a panel waits for its first data. */
export function Waiting({ children }: { children: ReactNode }) {
  return <div className="waiting">{children}</div>;
}
