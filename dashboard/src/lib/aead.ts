/**
 * Helpers for the AEAD demo wire format (DESIGN 3.3):
 * `base64url(nonce(24) || ciphertext || tag(16))`, XChaCha20-Poly1305.
 * The browser never encrypts or decrypts - the server does - so the only
 * operation here is the deliberate corruption used by the tamper button.
 */

import { parseBase64, toBase64Url } from './bytes.ts';

export const NONCE_BYTES = 24;
export const TAG_BYTES = 16;

export interface Tampered {
  ciphertext: string;
  /** Byte offset (in the decoded wire bytes) that was modified. */
  offset: number;
}

/**
 * Flip the lowest bit of one ciphertext byte (never the nonce). Poly1305
 * authenticates every byte, so decryption must now fail as a whole - there
 * is no "slightly wrong" plaintext to recover.
 */
export function tamperOneByte(wire: string): Tampered | null {
  const bytes = parseBase64(wire);
  if (!bytes || bytes.length < NONCE_BYTES + TAG_BYTES) return null;
  const offset = NONCE_BYTES + Math.floor((bytes.length - NONCE_BYTES) / 2);
  const copy = bytes.slice();
  copy[offset] = (copy[offset] ?? 0) ^ 0x01;
  return { ciphertext: toBase64Url(copy), offset };
}
