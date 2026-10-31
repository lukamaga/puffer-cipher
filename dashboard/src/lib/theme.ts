/**
 * Theme preference: follow the OS ("auto") or pin light/dark.
 *
 * The choice lives in localStorage as a per-viewer convenience only; every
 * access is guarded because storage can be unavailable (private windows,
 * blocked site data). index.html applies the stored value before first
 * paint to avoid a flash of the wrong theme.
 */

export type ThemePreference = 'auto' | 'light' | 'dark';

const KEY = 'puffer-theme';
export const THEME_ORDER: ThemePreference[] = ['auto', 'light', 'dark'];

export function readTheme(): ThemePreference {
  try {
    const value = localStorage.getItem(KEY);
    return value === 'light' || value === 'dark' ? value : 'auto';
  } catch {
    return 'auto';
  }
}

export function applyTheme(theme: ThemePreference): void {
  const root = document.documentElement;
  if (theme === 'auto') delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === 'auto') localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, theme);
  } catch {
    // Preference simply won't persist.
  }
  window.dispatchEvent(new Event('puffer-theme'));
}

/**
 * Resolve CSS custom properties to concrete colours for canvas drawing.
 * Custom properties holding light-dark() are not resolved by
 * getComputedStyle, so each is applied to a probe element's `color`.
 */
export function resolveColors<K extends string>(names: readonly K[], scope: Element): Record<K, string> {
  const probe = document.createElement('span');
  probe.style.display = 'none';
  scope.appendChild(probe);
  const out = {} as Record<K, string>;
  for (const name of names) {
    probe.style.color = `var(--${name})`;
    out[name] = getComputedStyle(probe).color;
  }
  probe.remove();
  return out;
}

/** Subscribe to anything that can change resolved colours: OS scheme or the toggle. */
export function onThemeChange(callback: () => void): () => void {
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  media.addEventListener('change', callback);
  window.addEventListener('puffer-theme', callback);
  return () => {
    media.removeEventListener('change', callback);
    window.removeEventListener('puffer-theme', callback);
  };
}
