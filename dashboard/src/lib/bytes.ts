/**
 * Byte and text encodings used on the wire (DESIGN.md section 3).
 *
 * Hex is the beacon's JSON encoding, base64url (no padding) is the AEAD demo's
 * encoding and plain base64 carries JPEG frames. Parsers return `null` instead
 * of throwing: every input here comes from the network, and the callers
 * (verifier, decoders) turn a malformed field into a precise, user-visible
 * failure rather than an exception.
 *
 * This module is shared by the browser bundle, the vitest suite and the Node
 * mock server, so it only uses APIs present in all three (`atob`, `btoa`,
 * `TextEncoder`).
 */

const HEX_RE = /^(?:[0-9a-fA-F]{2})*$/;

/** Decode hex (either case). Returns `null` on bad characters or on a length mismatch. */
export function parseHex(hex: string, byteLength?: number): Uint8Array | null {
  if (!HEX_RE.test(hex)) return null;
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = Number.parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  }
  if (byteLength !== undefined && out.length !== byteLength) return null;
  return out;
}

/** Lowercase hex, the canonical form used by the Python side. */
export function toHex(bytes: Uint8Array): string {
  let s = '';
  for (const b of bytes) s += b.toString(16).padStart(2, '0');
  return s;
}

/** Decode standard or URL-safe base64, with or without padding. */
export function parseBase64(text: string): Uint8Array | null {
  const normalized = text.replace(/-/g, '+').replace(/_/g, '/');
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(normalized) || normalized.length % 4 === 1) return null;
  const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, '=');
  try {
    const binary = atob(padded);
    const out = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  } catch {
    return null;
  }
}

/** base64url without padding (RFC 4648 section 5), the AEAD demo wire format. */
export function toBase64Url(bytes: Uint8Array): string {
  let binary = '';
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

const encoder = new TextEncoder();

export function utf8(text: string): Uint8Array {
  return encoder.encode(text);
}

export function concatBytes(...parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.length;
  }
  return out;
}

/**
 * Byte equality. Not constant-time on purpose: it only ever compares public
 * values (beacon hashes, public keys), never secrets.
 */
export function bytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

export function isAllZero(bytes: Uint8Array): boolean {
  return bytes.every((b) => b === 0);
}
