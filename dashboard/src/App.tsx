/**
 * Dashboard shell: one live client and one entropy history for the whole
 * page, shared through context/props, and a CSS-grid layout whose source
 * order is also the reading order on a phone.
 */

import { useEffect, useMemo } from 'react';

import { AquariumPanel } from './components/AquariumPanel.tsx';
import { BeaconPanel } from './components/BeaconPanel.tsx';
import { BitsPanel } from './components/BitsPanel.tsx';
import { EncryptPanel } from './components/EncryptPanel.tsx';
import { EntropyPanel } from './components/EntropyPanel.tsx';
import { FishPanel, LeaderboardPanel, useFishCards } from './components/FishPanel.tsx';
import { Header } from './components/Header.tsx';
import { HealthPanel } from './components/HealthPanel.tsx';
import { KpiRow } from './components/KpiRow.tsx';
import { LivenessPanel } from './components/LivenessPanel.tsx';
import { EntropyHistory } from './lib/history.ts';
import { LiveContext, useConnectionStatus, useLive } from './lib/live.ts';
import { LiveClient, liveUrl } from './lib/ws.ts';

function Dashboard({ history }: { history: EntropyHistory }) {
  const status = useConnectionStatus();
  const hello = useLive('hello');
  const fish = useFishCards();
  // While disconnected, panels keep their last render at reduced opacity
  // instead of blanking: stale data is labelled, not hidden.
  const stale = status.state !== 'open';
  return (
    <>
      <Header />
      <main className="layout" data-stale={stale || undefined}>
        <KpiRow history={history} fish={fish} />
        <AquariumPanel />
        <div className="area-side">
          <LivenessPanel />
          <HealthPanel />
        </div>
        <EntropyPanel history={history} />
        <BitsPanel />
        <FishPanel fish={fish} />
        <LeaderboardPanel fish={fish} />
        <EncryptPanel fish={fish} />
        <BeaconPanel />
      </main>
      <footer className="footer">
        <p>
          <strong>Data source: {hello?.source ?? 'unknown'}.</strong> Every secret is{' '}
          <span className="mono">BLAKE2b(key = os.urandom(32), pool)</span>, so output is never weaker than
          the operating system: the fish only add entropy.
        </p>
        <p className="muted">puffer-cipher · an aquarium-powered successor to LavaRand</p>
      </footer>
    </>
  );
}

export function App() {
  const client = useMemo(() => new LiveClient({ url: liveUrl(window.location) }), []);
  const history = useMemo(() => new EntropyHistory(300), []);

  useEffect(() => {
    client.start();
    return () => client.stop();
  }, [client]);
  useEffect(() => client.on('entropy', (data) => history.push(data)), [client, history]);

  return (
    <LiveContext.Provider value={client}>
      <Dashboard history={history} />
    </LiveContext.Provider>
  );
}
