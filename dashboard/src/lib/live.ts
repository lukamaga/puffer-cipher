/**
 * React bindings for the live feed and REST polling.
 *
 * High-rate streams (frames at 10 Hz, bits at 5 Hz) are consumed by canvas
 * components through `client.on` directly and never touch React state;
 * these hooks are for the slower, render-worthy messages.
 */

import {
  createContext,
  type RefCallback,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

import type { MessageMap, MessageType } from './protocol.ts';
import type { ConnectionStatus, LiveClient } from './ws.ts';

export const LiveContext = createContext<LiveClient | null>(null);

export function useLiveClient(): LiveClient {
  const client = useContext(LiveContext);
  if (!client) throw new Error('useLiveClient must be used inside <LiveContext.Provider>');
  return client;
}

/** Latest payload of one message type (re-renders on every message of that type). */
export function useLive<T extends MessageType>(type: T): MessageMap[T] | undefined {
  const client = useLiveClient();
  const subscribe = useCallback((notify: () => void) => client.on(type, notify), [client, type]);
  return useSyncExternalStore(subscribe, () => client.latest(type));
}

export function useConnectionStatus(): ConnectionStatus {
  const client = useLiveClient();
  const subscribe = useCallback((notify: () => void) => client.onStatus(notify), [client]);
  return useSyncExternalStore(subscribe, () => client.status);
}

export interface Polled<T> {
  data: T | undefined;
  error: Error | null;
}

/**
 * Fetch on mount and every `intervalMs` (0 = once). A failed refresh keeps
 * the previous data visible and reports the error alongside it.
 */
export function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number): Polled<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<Error | null>(null);
  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });
  useEffect(() => {
    let cancelled = false;
    const run = () => {
      fetcherRef.current().then(
        (value) => {
          if (cancelled) return;
          setData(value);
          setError(null);
        },
        (e: unknown) => {
          if (!cancelled) setError(e instanceof Error ? e : new Error(String(e)));
        },
      );
    };
    run();
    const timer = intervalMs > 0 ? setInterval(run, intervalMs) : null;
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [intervalMs]);

  return { data, error };
}

/** Wall-clock milliseconds, refreshed every `periodMs` (for countdowns and "ago" labels). */
export function useNow(periodMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), periodMs);
    return () => clearInterval(timer);
  }, [periodMs]);
  return now;
}

/** Content-box width of an element, tracked with ResizeObserver. */
export function useElementWidth<E extends Element>(): [RefCallback<E>, number] {
  const [width, setWidth] = useState(0);
  const observer = useRef<ResizeObserver | null>(null);
  const ref = useCallback((node: E | null) => {
    observer.current?.disconnect();
    if (!node) return;
    observer.current = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    observer.current.observe(node);
  }, []);
  return [ref, width];
}
