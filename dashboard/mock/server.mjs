// @ts-check
/**
 * Mock puffer backend: the REST API and `/ws/live` feed of DESIGN.md
 * section 7, served from a simulated aquarium so the dashboard can be
 * developed, demoed and screenshotted without Python, a camera or fish.
 *
 *   npm run mock          # http://localhost:8000, ws://localhost:8000/ws/live
 *   npm run dev           # Vite proxies /api and /ws here
 *
 * Fidelity where it matters:
 * - Beacon pulses are genuinely signed (Ed25519, deterministic mock key) and
 *   hash-chained, so the browser verifier does real work; no special cases.
 * - The encrypt demo uses XChaCha20-Poly1305 (@noble/ciphers) with the real
 *   wire format `base64url(nonce || ciphertext)` and aad "puffer-msg-v1", so
 *   the tamper button fails authentication for the right reason.
 * - Frames carry tracks but no JPEG; the dashboard renders a placeholder.
 *
 * Mock-only convenience: on connect it replays the last five minutes of
 * `entropy` messages so charts are populated immediately (the real backend
 * streams live data only). History is produced by fast-forwarding the
 * simulation at startup.
 *
 * Environment: PORT (8000), MOCK_BEACON_PERIOD_S (10),
 * MOCK_DAYLIGHT = cycle | day | night (cycle), MOCK_SEED (7).
 */

import { randomBytes } from 'node:crypto';
import { createServer } from 'node:http';

import { xchacha20poly1305 } from '@noble/ciphers/chacha.js';
import { blake2b } from '@noble/hashes/blake2.js';
import { WebSocket, WebSocketServer } from 'ws';

import { parseBase64, toBase64Url, toHex, utf8 } from '../src/lib/bytes.ts';
import { MockAquarium } from './aquarium.ts';
import { BeaconChain, seedFromLabel } from './beacon-chain.ts';

/** @typedef {import('node:http').IncomingMessage} Request */
/** @typedef {import('node:http').ServerResponse} Response */
/** @typedef {import('../src/lib/protocol.ts').MessageMap} MessageMap */

const PORT = Number(process.env.PORT ?? 8000);
const BEACON_PERIOD_MS = Math.round(Number(process.env.MOCK_BEACON_PERIOD_S ?? 10) * 1000);
const SEED = Number(process.env.MOCK_SEED ?? 7);
const DAYLIGHT = parseDaylight(process.env.MOCK_DAYLIGHT ?? 'cycle');
const FPS = 10;
const HISTORY_S = 300;
const AAD = utf8('puffer-msg-v1');
const KEY_ID_PERSON = personalization('puffer-key-id');
const MAX_PLAINTEXT = 16 * 1024;
/** Drop video frames for a client whose socket buffer is this far behind. */
const MAX_BUFFERED = 1 << 20;

/**
 * @param {string} value
 * @returns {'day' | 'night' | 'cycle'}
 */
function parseDaylight(value) {
  if (value === 'day' || value === 'night' || value === 'cycle') return value;
  throw new Error(`MOCK_DAYLIGHT must be day, night or cycle, got ${value}`);
}

/**
 * hashlib's `person=` zero-pads to 16 bytes; @noble requires exactly 16.
 * @param {string} label
 */
function personalization(label) {
  const out = new Uint8Array(16);
  out.set(utf8(label));
  return out;
}

// --- simulation and beacon --------------------------------------------------

const aquarium = new MockAquarium({ width: 640, height: 360, fps: FPS, seed: SEED, daylight: DAYLIGHT });
const chain = new BeaconChain({
  seed: seedFromLabel('puffer-mock-beacon'),
  periodMs: BEACON_PERIOD_MS,
  randomBytes: (n) => new Uint8Array(randomBytes(n)),
});
const bootWallMs = Date.now();
/** @type {MessageMap['entropy'][]} */
const entropyHistory = [];

function recordEntropy() {
  entropyHistory.push(aquarium.entropy());
  if (entropyHistory.length > HISTORY_S * 2) entropyHistory.shift();
}

/** @param {number} timestampMs */
function publishPulse(timestampMs) {
  const receipt = aquarium.draw('beacon', 64);
  return chain.append(aquarium.beaconSummary(receipt), timestampMs);
}

// Fast-forward five minutes - exactly one day/night cycle - so charts, stats
// and the beacon chain have history (including one night) and the live view
// starts at dawn.
{
  const frames = HISTORY_S * FPS;
  const framesPerPulse = Math.max(1, Math.round((BEACON_PERIOD_MS / 1000) * FPS));
  for (let i = 1; i <= frames; i++) {
    aquarium.step();
    if (i % 5 === 0) recordEntropy();
    if (i % framesPerPulse === 0) publishPulse(bootWallMs - ((frames - i) / FPS) * 1000);
  }
}

// --- WebSocket feed -------------------------------------------------------------

/**
 * @template {keyof MessageMap} T
 * @param {T} type
 * @param {MessageMap[T]} data
 */
const encode = (type, data) => JSON.stringify({ type, data });

const server = createServer((req, res) => {
  handle(req, res).catch((error) => {
    console.error(error);
    sendJson(res, 500, { detail: 'internal error' });
  });
});
const wss = new WebSocketServer({ server, path: '/ws/live' });

/**
 * @param {string} message
 * @param {{ droppable?: boolean }} [options]
 */
function broadcast(message, options = {}) {
  for (const client of wss.clients) {
    if (client.readyState !== WebSocket.OPEN) continue;
    if (options.droppable && client.bufferedAmount > MAX_BUFFERED) continue;
    client.send(message);
  }
}

wss.on('connection', (socket) => {
  socket.send(
    encode('hello', {
      version: '0.1.0-mock',
      source: 'mock simulator',
      width: aquarium.width,
      height: aquarium.height,
      fps: FPS,
      beacon_public_key: chain.publicKeyHex,
    }),
  );
  for (const entropy of entropyHistory) socket.send(encode('entropy', entropy));
  socket.send(encode('health', aquarium.health()));
  socket.send(encode('led', aquarium.led()));
  socket.send(encode('fish', { fish: aquarium.fishCards() }));
});

const frameTimer = setInterval(() => {
  aquarium.step();
  const n = aquarium.frameIndex;
  broadcast(encode('frame', aquarium.frame()), { droppable: true });
  if (n % 2 === 0) broadcast(encode('bits', aquarium.bits()));
  if (n % 5 === 0) {
    recordEntropy();
    broadcast(encode('entropy', entropyHistory[entropyHistory.length - 1] ?? aquarium.entropy()));
    broadcast(encode('led', aquarium.led()));
  }
  if (n % 10 === 0) broadcast(encode('health', aquarium.health()));
  if (n % 50 === 0) broadcast(encode('fish', { fish: aquarium.fishCards() }));
}, 1000 / FPS);

const beaconTimer = setInterval(() => {
  broadcast(encode('beacon', publishPulse(Date.now())));
}, BEACON_PERIOD_MS);

// --- REST -----------------------------------------------------------------------

/** An error with an HTTP status, rendered as FastAPI-style `{detail}`. */
class HttpError extends Error {
  /**
   * @param {number} status
   * @param {string} detail
   */
  constructor(status, detail) {
    super(detail);
    this.status = status;
  }
}

/**
 * @param {Response} res
 * @param {number} status
 * @param {unknown} body
 */
function sendJson(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(body));
}

/**
 * @param {Request} req
 * @returns {Promise<Record<string, unknown>>}
 */
async function readJson(req) {
  let size = 0;
  /** @type {Buffer[]} */
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > 64 * 1024) throw new HttpError(413, 'request body too large');
    chunks.push(chunk);
  }
  try {
    const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    if (typeof body === 'object' && body !== null && !Array.isArray(body)) return body;
  } catch {
    // fall through
  }
  throw new HttpError(422, 'body must be a JSON object');
}

/**
 * @param {unknown} value
 * @param {string} name
 */
function requireString(value, name) {
  if (typeof value !== 'string') throw new HttpError(422, `${name} must be a string`);
  return value;
}

/**
 * @param {URLSearchParams} query
 * @param {string} name
 * @param {number} fallback
 * @param {number} min
 * @param {number} max
 */
function intParam(query, name, fallback, min, max) {
  const raw = query.get(name);
  const value = raw === null ? fallback : Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new HttpError(422, `${name} must be an integer in [${min}, ${max}]`);
  }
  return value;
}

/**
 * @param {Request} req
 * @param {Response} res
 */
async function handle(req, res) {
  const url = new URL(req.url ?? '/', 'http://mock');
  const route = `${req.method} ${url.pathname}`;
  try {
    const pulseMatch = /^\/api\/beacon\/pulse\/(\d+)$/.exec(url.pathname);
    if (req.method === 'GET' && pulseMatch) {
      const pulse = chain.pulses[Number(pulseMatch[1])];
      return pulse ? sendJson(res, 200, pulse) : sendJson(res, 404, { detail: 'no such pulse' });
    }
    switch (route) {
      case 'GET /api/status':
        return sendJson(res, 200, aquarium.status((Date.now() - bootWallMs) / 1000 + HISTORY_S));
      case 'GET /api/random': {
        const n = intParam(url.searchParams, 'n', 32, 1, 1024);
        const format = url.searchParams.get('format') ?? 'hex';
        if (format !== 'hex' && format !== 'base64') throw new HttpError(422, 'format must be hex or base64');
        const bytes = new Uint8Array(randomBytes(n));
        const data = format === 'hex' ? toHex(bytes) : Buffer.from(bytes).toString('base64');
        return sendJson(res, 200, { data, receipt: aquarium.draw('random', n) });
      }
      case 'POST /api/demo/encrypt': {
        const plaintext = requireString((await readJson(req)).plaintext, 'plaintext');
        if (utf8(plaintext).length > MAX_PLAINTEXT)
          throw new HttpError(413, 'plaintext is limited to 16 KiB');
        const receipt = aquarium.draw('demo-key', 32);
        const key = new Uint8Array(randomBytes(32));
        const nonce = new Uint8Array(randomBytes(24));
        const sealed = xchacha20poly1305(key, nonce, AAD).encrypt(utf8(plaintext));
        const wire = new Uint8Array(nonce.length + sealed.length);
        wire.set(nonce);
        wire.set(sealed, nonce.length);
        return sendJson(res, 200, {
          ciphertext: toBase64Url(wire),
          key: toBase64Url(key),
          key_id: toHex(blake2b(key, { dkLen: 8, personalization: KEY_ID_PERSON })),
          receipt,
        });
      }
      case 'POST /api/demo/decrypt': {
        const body = await readJson(req);
        const wire = parseBase64(requireString(body.ciphertext, 'ciphertext'));
        const key = parseBase64(requireString(body.key, 'key'));
        if (!key || key.length !== 32) throw new HttpError(400, 'key must be 32 bytes of base64url');
        if (!wire || wire.length < 24 + 16) throw new HttpError(400, 'ciphertext is too short');
        try {
          const opened = xchacha20poly1305(key, wire.subarray(0, 24), AAD).decrypt(wire.subarray(24));
          return sendJson(res, 200, { plaintext: new TextDecoder('utf-8', { fatal: true }).decode(opened) });
        } catch {
          throw new HttpError(400, 'authentication failed: the ciphertext was modified or the key is wrong');
        }
      }
      case 'GET /api/fish':
        return sendJson(res, 200, aquarium.fishCards());
      case 'GET /api/leaderboard':
        return sendJson(res, 200, aquarium.leaderboard(intParam(url.searchParams, 'days', 7, 1, 365)));
      case 'GET /api/beacon/public-key':
        return sendJson(res, 200, { algorithm: 'Ed25519', public_key: chain.publicKeyHex });
      case 'GET /api/beacon/latest':
        return chain.latest
          ? sendJson(res, 200, chain.latest)
          : sendJson(res, 404, { detail: 'no pulses yet' });
      case 'GET /api/beacon/pulses': {
        const start = intParam(url.searchParams, 'start', 0, 0, Number.MAX_SAFE_INTEGER);
        const limit = intParam(url.searchParams, 'limit', 50, 1, 500);
        return sendJson(res, 200, chain.pulses.slice(start, start + limit));
      }
      default:
        return sendJson(res, 404, { detail: `no route for ${route}` });
    }
  } catch (error) {
    if (error instanceof HttpError) return sendJson(res, error.status, { detail: error.message });
    throw error;
  }
}

server.listen(PORT, () => {
  console.log(`puffer mock backend on http://localhost:${PORT}  (ws /ws/live)`);
  console.log(`beacon key ${chain.publicKeyHex}, ${chain.pulses.length} pulses of history`);
});

function shutdown() {
  clearInterval(frameTimer);
  clearInterval(beaconTimer);
  for (const client of wss.clients) client.terminate();
  wss.close();
  server.close(() => process.exit(0));
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
