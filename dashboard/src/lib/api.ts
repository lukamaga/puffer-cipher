/**
 * Typed REST client for `/api` (DESIGN.md 7.1).
 *
 * Every response body goes through the matching decoder from `protocol.ts`,
 * so components receive either a well-typed value or an `ApiError` - never a
 * half-shaped object. Paths are relative: the dashboard is always served from
 * the same origin as the API (FastAPI in production, the Vite proxy in dev).
 */

import {
  type BeaconPulse,
  type Decoder,
  type DecryptRequest,
  type DecryptResponse,
  type EncryptRequest,
  type EncryptResponse,
  type FishCard,
  type LeaderboardEntry,
  type PublicKeyResponse,
  restDecoders,
  type StatusResponse,
} from './protocol.ts';

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** FastAPI errors are `{detail: string}` or, for validation, `{detail: [{msg}, ...]}`. */
function errorDetail(body: unknown, fallback: string): string {
  if (typeof body !== 'object' || body === null || !('detail' in body)) return fallback;
  const { detail } = body as { detail: unknown };
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((d: unknown) =>
        typeof d === 'object' && d !== null && 'msg' in d ? String((d as { msg: unknown }).msg) : '',
      )
      .filter(Boolean);
    if (messages.length > 0) return messages.join('; ');
  }
  return fallback;
}

async function request<T>(path: string, decoder: Decoder<T>, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { Accept: 'application/json', ...init?.headers },
  });
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    // Non-JSON body (proxy error page, empty 204): handled below.
  }
  if (!response.ok) {
    throw new ApiError(response.status, errorDetail(body, `${response.status} ${response.statusText}`));
  }
  try {
    return decoder(body);
  } catch (e) {
    throw new ApiError(response.status, `unexpected response from ${path}: ${String(e)}`);
  }
}

function post<T>(path: string, payload: unknown, decoder: Decoder<T>): Promise<T> {
  return request(path, decoder, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export const api = {
  status: (): Promise<StatusResponse> => request('/status', restDecoders.status),
  fish: (): Promise<FishCard[]> => request('/fish', restDecoders.fish),
  leaderboard: (days = 7): Promise<LeaderboardEntry[]> =>
    request(`/leaderboard?days=${days}`, restDecoders.leaderboard),
  encrypt: (plaintext: string): Promise<EncryptResponse> =>
    post('/demo/encrypt', { plaintext } satisfies EncryptRequest, restDecoders.encrypt),
  decrypt: (ciphertext: string, key: string): Promise<DecryptResponse> =>
    post('/demo/decrypt', { ciphertext, key } satisfies DecryptRequest, restDecoders.decrypt),
  beaconPublicKey: (): Promise<PublicKeyResponse> => request('/beacon/public-key', restDecoders.publicKey),
  beaconLatest: (): Promise<BeaconPulse> => request('/beacon/latest', restDecoders.pulse),
  beaconPulses: (start: number, limit: number): Promise<BeaconPulse[]> =>
    request(`/beacon/pulses?start=${start}&limit=${limit}`, restDecoders.pulses),
};
