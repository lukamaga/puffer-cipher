import { useEffect, useRef, useState } from 'react';

import { renderScene } from '../lib/aquariumRender.ts';
import { parseBase64 } from '../lib/bytes.ts';
import { useConnectionStatus, useLive, useLiveClient } from '../lib/live.ts';
import type { FrameData } from '../lib/protocol.ts';
import { Icon, Panel } from './ui.tsx';

interface TrackSummary {
  id: number;
  name: string | null;
  color: string | null;
  speed: number;
}

interface Readout {
  fps: number;
  frame: number;
  tracks: TrackSummary[];
  hasVideo: boolean;
}

async function decodeJpeg(base64: string | null): Promise<ImageBitmap | null> {
  if (!base64) return null;
  const bytes = parseBase64(base64);
  if (!bytes) return null;
  try {
    return await createImageBitmap(new Blob([bytes as BlobPart], { type: 'image/jpeg' }));
  } catch {
    return null; // a corrupt frame is skipped, the overlay still draws
  }
}

/**
 * Live aquarium: the latest frame with tracker overlays. Frames are drawn
 * straight from the socket callback (no React state per frame); a frame that
 * arrives while the previous JPEG is still decoding replaces any queued one,
 * so a slow device shows fewer frames rather than falling behind.
 */
export function AquariumPanel() {
  const client = useLiveClient();
  const hello = useLive('hello');
  const entropy = useLive('entropy');
  const status = useConnectionStatus();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const modeRef = useRef(entropy?.mode ?? 'day');
  const [readout, setReadout] = useState<Readout | null>(null);

  useEffect(() => {
    modeRef.current = entropy?.mode ?? 'day';
  }, [entropy?.mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    let latest: FrameData | null = null;
    let busy = false;
    let lastArrival = 0;
    let fps = 0;
    let lastReadout = 0;
    const started = performance.now();

    const fit = (frameWidth: number, frameHeight: number) => {
      const cssWidth = canvas.clientWidth || frameWidth;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.round(cssWidth * dpr);
      const height = Math.round((cssWidth * frameHeight * dpr) / frameWidth);
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      return cssWidth;
    };

    const pump = async () => {
      if (busy || !latest) return;
      busy = true;
      const frame = latest;
      latest = null;
      const image = await decodeJpeg(frame.jpeg);
      const cssWidth = fit(frame.width, frame.height);
      renderScene(ctx, cssWidth, {
        width: frame.width,
        height: frame.height,
        tracks: frame.tracks,
        image,
        mode: modeRef.current,
        time: (performance.now() - started) / 1000,
      });
      image?.close();
      const now = performance.now();
      if (now - lastReadout > 500) {
        lastReadout = now;
        setReadout({
          fps,
          frame: frame.index,
          tracks: frame.tracks
            .map((t) => ({ id: t.track_id, name: t.name, color: t.color, speed: t.speed }))
            .sort((a, b) => (a.name ?? '~').localeCompare(b.name ?? '~')),
          hasVideo: image !== null,
        });
      }
      busy = false;
      if (latest) void pump();
    };

    const off = client.on('frame', (frame) => {
      const now = performance.now();
      if (lastArrival > 0) {
        const instant = 1000 / Math.max(1, now - lastArrival);
        fps = fps === 0 ? instant : fps + (instant - fps) * 0.1;
      }
      lastArrival = now;
      latest = frame;
      void pump();
    });
    return off;
  }, [client]);

  const width = hello?.width ?? 640;
  const height = hello?.height ?? 360;
  const live = status.state === 'open' && readout !== null;
  const night = entropy?.mode === 'night';

  return (
    <Panel
      className="area-aquarium"
      title="Live aquarium"
      subtitle="Tracked fish with their recent paths. The motion the Kalman filter could not predict is what the fish channel turns into entropy."
      actions={
        readout && (
          <span className="chip chip--quiet">
            {readout.tracks.filter((t) => t.name !== null).length}/{readout.tracks.length} identified
          </span>
        )
      }
    >
      <div className="viewport" style={{ aspectRatio: `${width} / ${height}` }}>
        <canvas
          ref={canvasRef}
          className="viewport__canvas"
          aria-label="Aquarium camera view with tracking overlays"
          role="img"
        />
        <div className="viewport__overlay viewport__overlay--tl">
          <span className={`hud${live ? ' hud--live' : ''}`}>
            <span className="hud__dot" aria-hidden />
            {live ? 'LIVE' : 'WAITING'}
          </span>
          {readout && <span className="hud hud--mono">frame {readout.frame.toLocaleString('en')}</span>}
        </div>
        <div className="viewport__overlay viewport__overlay--tr">
          {readout && <span className="hud hud--mono">{readout.fps.toFixed(1)} fps</span>}
          <span className="hud">
            <Icon name={night ? 'moon' : 'sun'} size={13} />
            {night ? 'Night · IR' : 'Day'}
          </span>
        </div>
        {readout && !readout.hasVideo && (
          <div className="viewport__overlay viewport__overlay--bl">
            <span className="hud hud--quiet">No video in stream · tracks only</span>
          </div>
        )}
        {!readout && (
          <div className="viewport__empty">
            {status.state === 'open' ? 'Waiting for the first frame…' : 'Connecting to the live feed…'}
          </div>
        )}
      </div>
      {readout && readout.tracks.length > 0 && (
        <ul className="tracks" aria-label="Current tracks">
          {readout.tracks.map((t) => (
            <li key={t.id} className="tracks__item">
              <span className="swatch" style={{ background: t.color ?? 'var(--neutral)' }} aria-hidden />
              <span className="tracks__name">{t.name ?? 'identifying…'}</span>
              <span className="tracks__meta">
                #{t.id} · {Math.round(t.speed)} px/s
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
