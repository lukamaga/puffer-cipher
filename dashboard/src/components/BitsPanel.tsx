import { useEffect, useRef, useState } from 'react';

import { useLiveClient } from '../lib/live.ts';
import { onThemeChange, resolveColors } from '../lib/theme.ts';
import { Panel, Waiting } from './ui.tsx';

const COLUMNS = 64;
const MESSAGES = 8; // 8 messages x 256 bits = 32 rows of 64
const ROWS_PER_MESSAGE = 256 / COLUMNS;

interface BitStats {
  ones: number;
  total: number;
  messages: number;
  latestHex: string;
}

function bitsToHex(bits: string): string {
  let hex = '';
  for (let i = 0; i + 4 <= bits.length; i += 4) hex += Number.parseInt(bits.slice(i, i + 4), 2).toString(16);
  return hex;
}

/**
 * A running matrix of the public display stream: newest message on top,
 * older ones fading out. It is there to make "randomness" visible and to let
 * a viewer eyeball gross bias - the ones-ratio below is a sanity readout, not
 * a statistical test (those run on raw samples, see the analysis report).
 */
export function BitsPanel() {
  const client = useLiveClient();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stats, setStats] = useState<BitStats | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const rows: string[] = [];
    let colors = resolveColors(['bit-on', 'bit-off'], canvas);
    let ones = 0;
    let total = 0;
    let messages = 0;

    const draw = () => {
      const cssWidth = canvas.clientWidth;
      if (cssWidth === 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cell = cssWidth / COLUMNS;
      const height = cell * MESSAGES * ROWS_PER_MESSAGE;
      canvas.width = Math.round(cssWidth * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssWidth, height);
      const gap = cell > 5 ? 1 : 0.5;
      rows.forEach((row, r) => {
        const age = Math.floor(r / ROWS_PER_MESSAGE);
        const fade = 1 - (age / MESSAGES) * 0.75;
        for (let c = 0; c < COLUMNS; c++) {
          const on = row.charCodeAt(c) === 49;
          ctx.globalAlpha = on ? fade : 1;
          ctx.fillStyle = on ? colors['bit-on'] : colors['bit-off'];
          ctx.beginPath();
          ctx.roundRect(
            c * cell + gap / 2,
            r * cell + gap / 2,
            cell - gap,
            cell - gap,
            Math.min(1.5, cell / 4),
          );
          ctx.fill();
        }
      });
      ctx.globalAlpha = 1;
    };

    const off = client.on('bits', ({ bits }) => {
      const fresh: string[] = [];
      for (let i = 0; i + COLUMNS <= bits.length; i += COLUMNS) fresh.push(bits.slice(i, i + COLUMNS));
      rows.unshift(...fresh);
      rows.length = Math.min(rows.length, MESSAGES * ROWS_PER_MESSAGE);
      for (const ch of bits) if (ch === '1') ones += 1;
      total += bits.length;
      messages += 1;
      draw();
      if (messages % 5 === 1) setStats({ ones, total, messages, latestHex: bitsToHex(bits) });
    });
    const resize = new ResizeObserver(draw);
    resize.observe(canvas);
    const offTheme = onThemeChange(() => {
      colors = resolveColors(['bit-on', 'bit-off'], canvas);
      draw();
    });
    return () => {
      off();
      resize.disconnect();
      offTheme();
    };
  }, [client]);

  const ratio = stats && stats.total > 0 ? stats.ones / stats.total : null;
  return (
    <Panel
      className="area-bits"
      title="Public bit stream"
      subtitle="256 bits per message from a dedicated display stream. Public by design - never used as key material."
    >
      <div className="bits">
        <canvas
          ref={canvasRef}
          className="bits__canvas"
          role="img"
          aria-label="Matrix of recent display bits; filled cells are ones"
        />
        {!stats && <Waiting>Waiting for bits…</Waiting>}
      </div>
      {stats && ratio !== null && (
        <dl className="kv-grid kv-grid--3">
          <div>
            <dt>Ones</dt>
            <dd>{(ratio * 100).toFixed(2)}%</dd>
          </div>
          <div>
            <dt>Bits seen</dt>
            <dd>{stats.total.toLocaleString('en')}</dd>
          </div>
          <div>
            <dt>Messages</dt>
            <dd>{stats.messages.toLocaleString('en')}</dd>
          </div>
          <div className="kv-grid__wide">
            <dt>Latest, hex</dt>
            <dd className="mono mono--wrap">{stats.latestHex}</dd>
          </div>
        </dl>
      )}
    </Panel>
  );
}
