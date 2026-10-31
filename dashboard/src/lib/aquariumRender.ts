/**
 * Canvas drawing for the live aquarium view.
 *
 * Everything is drawn in *frame* pixel coordinates (the space the tracker's
 * boxes and trails live in) through one transform, so overlays stay aligned
 * with the video at any canvas size. Stroke widths and fonts are divided by
 * the scale to stay constant in CSS pixels.
 *
 * The overlay is drawn in the same pass as the image it belongs to: boxes
 * are never interpolated or drawn over a different frame, because a box
 * that drifts off its fish would misrepresent what the tracker measured.
 *
 * The video area is not themed - it is a camera image - so its colours are
 * constants rather than CSS tokens.
 */

import type { FrameTrack, LightMode } from './protocol.ts';

export interface SceneFrame {
  width: number;
  height: number;
  tracks: FrameTrack[];
  image: ImageBitmap | null;
  mode: LightMode;
  /** Animation time in seconds, for the placeholder water. */
  time: number;
}

const UNIDENTIFIED = '#c5d3da';
const LABEL_BG = 'rgba(3, 16, 23, 0.74)';
const LABEL_FONT = '600 12px system-ui, -apple-system, "Segoe UI", sans-serif';
const ID_FONT = '500 11px ui-monospace, "SF Mono", Menlo, Consolas, monospace';

/** Deterministic pseudo-random sequence for placeholder particles. */
function hash01(n: number): number {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

/**
 * Placeholder water for frames without an image: depth gradient, slow light
 * shafts, drifting particles and a vignette. Night mode switches to the
 * monochrome look of an IR camera.
 */
function drawWater(ctx: CanvasRenderingContext2D, w: number, h: number, mode: LightMode, time: number) {
  const night = mode === 'night';
  const depth = ctx.createLinearGradient(0, 0, 0, h);
  if (night) {
    depth.addColorStop(0, '#1d2a2b');
    depth.addColorStop(0.55, '#121c1d');
    depth.addColorStop(1, '#0a1112');
  } else {
    depth.addColorStop(0, '#11607a');
    depth.addColorStop(0.45, '#0b3d52');
    depth.addColorStop(1, '#061f2c');
  }
  ctx.fillStyle = depth;
  ctx.fillRect(0, 0, w, h);

  // Light shafts from the surface, swaying very slowly.
  ctx.save();
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 0; i < 5; i++) {
    const base = w * (0.12 + i * 0.2) + Math.sin(time * 0.15 + i * 1.7) * w * 0.03;
    const spread = w * (0.05 + 0.03 * hash01(i));
    const ray = ctx.createLinearGradient(0, 0, 0, h * 0.95);
    const alpha = night ? 0.025 : 0.06 + 0.03 * hash01(i + 9);
    ray.addColorStop(0, `rgba(190, 240, 255, ${alpha})`);
    ray.addColorStop(1, 'rgba(190, 240, 255, 0)');
    ctx.fillStyle = ray;
    ctx.beginPath();
    ctx.moveTo(base - spread * 0.4, 0);
    ctx.lineTo(base + spread * 0.4, 0);
    ctx.lineTo(base + spread * 1.6 + w * 0.06, h);
    ctx.lineTo(base - spread * 0.2 + w * 0.06, h);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();

  // Sand / gravel floor.
  const floor = ctx.createLinearGradient(0, h * 0.82, 0, h);
  floor.addColorStop(0, night ? 'rgba(60, 72, 70, 0)' : 'rgba(150, 140, 110, 0)');
  floor.addColorStop(1, night ? 'rgba(60, 72, 70, 0.35)' : 'rgba(150, 140, 110, 0.28)');
  ctx.fillStyle = floor;
  ctx.fillRect(0, h * 0.82, w, h * 0.18);

  // Marine snow.
  ctx.fillStyle = night ? 'rgba(210, 230, 225, 0.22)' : 'rgba(220, 245, 255, 0.28)';
  for (let i = 0; i < 46; i++) {
    const x = (hash01(i) * w + time * (2 + 4 * hash01(i + 40))) % w;
    const y = (hash01(i + 90) * h + time * (1.5 + 3 * hash01(i + 7))) % h;
    const r = 0.6 + 1.1 * hash01(i + 13);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  // Vignette.
  const vignette = ctx.createRadialGradient(w / 2, h * 0.45, h * 0.3, w / 2, h * 0.5, w * 0.72);
  vignette.addColorStop(0, 'rgba(0, 0, 0, 0)');
  vignette.addColorStop(1, 'rgba(0, 8, 12, 0.5)');
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, w, h);
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, Math.min(r, w / 2, h / 2));
}

function drawTrail(ctx: CanvasRenderingContext2D, track: FrameTrack, color: string, px: number) {
  const points = track.trail;
  if (points.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * px;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  // Fade from transparent (oldest) to strong (newest), segment by segment.
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    if (!a || !b) continue;
    ctx.globalAlpha = 0.85 * (i / (points.length - 1)) ** 1.6;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

function drawBox(ctx: CanvasRenderingContext2D, track: FrameTrack, color: string, px: number) {
  const [x1, y1, x2, y2] = track.bbox;
  const w = x2 - x1;
  const h = y2 - y1;
  ctx.save();
  // A faint wash marks the measured object without pretending to be the fish.
  ctx.globalAlpha = 0.1;
  ctx.fillStyle = color;
  roundRect(ctx, x1, y1, w, h, 6 * px);
  ctx.fill();
  ctx.globalAlpha = 0.95;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6 * px;
  roundRect(ctx, x1, y1, w, h, 6 * px);
  ctx.stroke();
  // Centre marker, ringed so it reads over any background.
  const cx = (x1 + x2) / 2;
  const cy = (y1 + y2) / 2;
  ctx.globalAlpha = 1;
  ctx.fillStyle = 'rgba(3, 16, 23, 0.65)';
  ctx.beginPath();
  ctx.arc(cx, cy, 4 * px, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(cx, cy, 2.4 * px, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const overlaps = (a: Rect, b: Rect) =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/**
 * Name tags, placed greedily: above the box, else below, else right-aligned,
 * avoiding tags already placed and the HUD band at the top. Fish school
 * together, and overlapping tags would hide exactly the identities a viewer
 * is trying to read. On small canvases tags drop the track id.
 */
function drawLabels(
  ctx: CanvasRenderingContext2D,
  tracks: FrameTrack[],
  px: number,
  frameWidth: number,
  compact: boolean,
) {
  const fontSize = compact ? 10.5 : 12;
  const labelFont = LABEL_FONT.replace('12px', `${fontSize * px}px`);
  const idFont = ID_FONT.replace('11px', `${11 * px}px`);
  const padX = (compact ? 5 : 7) * px;
  const dot = (compact ? 6 : 7) * px;
  const gap = (compact ? 4 : 6) * px;
  const height = (compact ? 17 : 20) * px;
  const margin = 4 * px;
  const placed: Rect[] = [{ x: 0, y: 0, w: frameWidth, h: 36 * px }];

  ctx.save();
  ctx.textBaseline = 'middle';
  for (const track of [...tracks].sort((a, b) => a.bbox[1] - b.bbox[1])) {
    const color = track.color ?? UNIDENTIFIED;
    const name = track.name ?? (compact ? '…' : 'identifying…');
    const id = compact ? '' : `#${track.track_id}`;
    ctx.font = labelFont;
    const nameWidth = ctx.measureText(name).width;
    ctx.font = idFont;
    const idWidth = id ? ctx.measureText(id).width + gap : 0;
    const width = padX + dot + gap + nameWidth + idWidth + padX;
    const [x1, y1, x2, y2] = track.bbox;
    const clampX = (x: number) => Math.max(0, Math.min(x, frameWidth - width));
    const candidates: Rect[] = [
      { x: clampX(x1), y: y1 - height - margin, w: width, h: height },
      { x: clampX(x1), y: y2 + margin, w: width, h: height },
      { x: clampX(x2 - width), y: y1 - height - margin, w: width, h: height },
      { x: clampX(x2 - width), y: y2 + margin, w: width, h: height },
    ];
    const box = candidates.find((c) => c.y >= 0 && !placed.some((p) => overlaps(c, p))) ?? candidates[1];
    if (!box) continue;
    placed.push(box);

    ctx.fillStyle = LABEL_BG;
    roundRect(ctx, box.x, box.y, box.w, box.h, box.h / 2);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(box.x + padX + dot / 2, box.y + height / 2, dot / 2, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = labelFont;
    ctx.fillStyle = track.name ? '#f2f8fa' : '#b9c8cf';
    ctx.fillText(name, box.x + padX + dot + gap, box.y + height / 2 + 0.5 * px);
    if (id) {
      ctx.font = idFont;
      ctx.fillStyle = 'rgba(214, 230, 236, 0.66)';
      ctx.fillText(id, box.x + padX + dot + gap + nameWidth + gap, box.y + height / 2 + 0.5 * px);
    }
  }
  ctx.restore();
}

/**
 * Render one frame onto a canvas whose backing store is `canvas.width` x
 * `canvas.height` device pixels and whose CSS width is `cssWidth`.
 */
export function renderScene(ctx: CanvasRenderingContext2D, cssWidth: number, scene: SceneFrame) {
  const canvas = ctx.canvas;
  const scale = canvas.width / scene.width;
  const px = scene.width / cssWidth; // one CSS pixel in frame units
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.clearRect(0, 0, scene.width, scene.height);
  if (scene.image) {
    ctx.drawImage(scene.image, 0, 0, scene.width, scene.height);
  } else {
    drawWater(ctx, scene.width, scene.height, scene.mode, scene.time);
  }
  const colorOf = (t: FrameTrack) => t.color ?? UNIDENTIFIED;
  for (const t of scene.tracks) drawTrail(ctx, t, colorOf(t), px);
  for (const t of scene.tracks) drawBox(ctx, t, colorOf(t), px);
  drawLabels(ctx, scene.tracks, px, scene.width, cssWidth < 520);
}
