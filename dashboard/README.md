# puffer-cipher dashboard

The live web UI for puffer-cipher: the aquarium with tracker overlays, per-channel entropy
credit, LED liveness, SP 800-90B health, the fish and their leaderboard, an encrypt/decrypt
demo with its entropy receipt, and a randomness beacon whose every pulse is **re-verified in
the browser** (Ed25519 + BLAKE2b).

React 19 + Vite + strict TypeScript, plain CSS, no UI framework and no charting library. The
only runtime dependencies besides React are `@noble/curves` and `@noble/hashes`, the audited
libraries that do the beacon cryptography.

## Running it

Requires Node **22.18+** (the mock server imports TypeScript modules natively).

```bash
npm ci

# Without the Python backend: realistic fake data on :8000
npm run mock          # REST /api/* and WebSocket /ws/live on http://localhost:8000
npm run dev           # http://localhost:5173 (proxies /api and /ws to :8000)

# With the real backend
puffer serve          # FastAPI on :8000, then:
npm run dev
```

`PUFFER_BACKEND=http://host:port npm run dev` points the dev proxy at another backend.

### Production build

```bash
npm run build         # type-checks, then writes dashboard/dist
npm run preview       # serves dist on :4173 with the same proxy
```

The FastAPI app serves `dist/` at `/`. Every request is same-origin (`/api/...`,
`/ws/live`), so the bundle needs no configuration.

## Scripts

| script                   | what it does                                                                |
| ------------------------ | --------------------------------------------------------------------------- |
| `npm run dev`            | Vite dev server with `/api` and `/ws` proxied to `:8000`                    |
| `npm run build`          | `tsc --noEmit` for the app, then the production bundle in `dist/`           |
| `npm run preview`        | serve the production build locally                                          |
| `npm test`               | Vitest: protocol decoding, reconnect logic, beacon verifier, shared vectors |
| `npm run lint`           | strict `tsc --noEmit` over the app, the tests, the Vite config and the mock |
| `npm run format[:check]` | Prettier                                                                    |
| `npm run mock`           | the mock backend (see below)                                                |

## Layout

```
src/lib/protocol.ts     types + runtime decoders for every REST payload and WS message (DESIGN 7)
src/lib/ws.ts           LiveClient: backoff with jitter, silence watchdog, typed subscriptions
src/lib/beacon.ts       260-byte pulse message, verification with the Python error codes (DESIGN 3.4)
src/lib/api.ts          typed REST client; every body is decoded before use
src/lib/*.ts            bytes/encodings, formatting, scales, theme, status vocabulary, hooks
src/components/         one component per panel, plus small primitives in ui.tsx
src/styles/             tokens.css (light-dark() design tokens), app.css
mock/server.mjs         mock backend: node:http + ws
mock/aquarium.ts        deterministic aquarium simulation emitting protocol types
mock/beacon-chain.ts    signs a real Ed25519/BLAKE2b hash chain (dev and tests only)
tests/                  Vitest suites and the libsodium-generated golden pulse
```

## Design decisions

- **Untrusted input is decoded at the boundary.** WebSocket and REST payloads pass through
  decoders typed as `Decoder<Interface>`, so the compiler proves the interfaces and the runtime
  checks agree. Unknown fields are dropped and unknown message types ignored (the backend can
  grow); a wrongly typed known field rejects the message, which is counted, never rendered.
- **The beacon is verified, not displayed.** The browser pins the key from the WebSocket
  handshake and checks every pulse itself: summary hash, signature (strict RFC 8032 mode, the
  closest match to libsodium), output, genesis, index, timestamp, chain link and precommitment.
  Error codes and their order match `puffer.beacon.pulse.ErrorCode`, so
  `tests/vectors/beacon_vectors.json` produces the same verdicts in pytest and Vitest. With no
  pinned key, a pulse is verified against its predecessor's key, so a chain cannot silently
  switch keys. If the server later serves a history that does not extend the pulses already
  seen (a reset or equivocation), the panel says so instead of merging the two.
- **Reconnects are boring.** Exponential backoff with jitter; the backoff resets only after a
  valid message (an accept-then-drop proxy cannot cause a tight loop); a watchdog replaces
  connections that go silent. While disconnected, panels keep their last data at reduced opacity.
- **High-rate data never touches React state.** Frames (10 Hz) and bits (5 Hz) are drawn to
  canvases straight from socket callbacks; a frame that arrives during JPEG decoding replaces
  the queued one, so slow devices drop frames instead of lagging.
- **Charts follow a validated palette.** Channel colours (sensor blue, fish orange, system
  aqua) are the first three slots of a reference categorical palette, validated for
  colour-vision deficiency and contrast against these exact surfaces in both themes. Entropy is
  drawn as small multiples with independent y-scales, because sensor noise earns ~10^4 bit/s
  and fish ~10^2 bit/s: one axis would flatten the fish, and a dual axis would invent a
  correlation. Every chart has a direct readout and a table view; status is always icon + label,
  never colour alone.
- **Light and dark themes** come from single-source `light-dark()` tokens. The OS preference is
  the default; the toggle pins a theme and wins in both directions.

## Mock backend

`npm run mock` serves the full protocol from a simulated aquarium (no camera, no Python):
fish with persistent identities whose track ids change when they hide and reappear, credit
from the DESIGN 4.3 formula, pools that saturate and reset on every draw, a Manchester LED
challenge with rare observation errors, a genuinely signed beacon chain (10 s period), and an
encrypt/decrypt demo using real XChaCha20-Poly1305 in the documented wire format. Frames carry
tracks but no JPEG; the canvas shows a placeholder and says so.

On start it fast-forwards five minutes (one day/night cycle) and replays that `entropy`
history to each new client, so charts are populated immediately. The real backend streams
live data only.

| variable               | default | meaning                                          |
| ---------------------- | ------- | ------------------------------------------------ |
| `PORT`                 | `8000`  | HTTP + WebSocket port                            |
| `MOCK_BEACON_PERIOD_S` | `10`    | beacon period                                    |
| `MOCK_DAYLIGHT`        | `cycle` | `cycle` (4 min day, 1 min night), `day`, `night` |
| `MOCK_SEED`            | `7`     | seed for the fish motion                         |

## Tests

`npm test` runs in well under a second:

- `protocol.test.ts`: every message type, malformed envelopes, field-level rejections with paths,
  REST payloads, encodings, the tamper helper, formatting.
- `ws.test.ts`: the reconnect state machine against a scripted fake socket and fake timers.
- `beacon.test.ts`: self-contained; pulses are assembled byte by byte and signed with @noble in
  the test, every error code is provoked, and a golden pulse computed with PyNaCl/libsodium
  anchors the layout to the Python implementation.
- `beacon-vectors.test.ts`: the cross-language vectors in `../tests/vectors/beacon_vectors.json`
  (skipped with a message when the file has not been generated).
