import { useState } from 'react';

import { api } from '../lib/api.ts';
import { type Tampered, tamperOneByte } from '../lib/aead.ts';
import { utf8 } from '../lib/bytes.ts';
import { CHANNEL_META } from '../lib/channels.ts';
import { formatBits, formatClock } from '../lib/format.ts';
import { useElementWidth } from '../lib/live.ts';
import {
  CAMERA_CHANNELS,
  type EncryptResponse,
  type EntropyReceipt,
  type FishCard,
} from '../lib/protocol.ts';
import { HEALTH_LOOK, LIVENESS_LOOK } from '../lib/status.ts';
import { CopyButton, Icon, Panel, StatusBadge } from './ui.tsx';

const MAX_PLAINTEXT_BYTES = 16 * 1024;
const LABEL_MIN_PX = 64;

type Decryption =
  | { kind: 'none' }
  | { kind: 'pending'; tampered: boolean }
  | { kind: 'ok'; plaintext: string }
  | { kind: 'failed'; detail: string; tampered: boolean };

function DiffText({ original, changed }: { original: string; changed: string }) {
  return (
    <>
      {[...changed].map((ch, i) =>
        ch === original[i] ? (
          ch
        ) : (
          <mark key={i} className="diff-mark">
            {ch}
          </mark>
        ),
      )}
    </>
  );
}

function Receipt({ receipt, fish }: { receipt: EntropyReceipt; fish: FishCard[] | undefined }) {
  const [barRef, barWidth] = useElementWidth<HTMLDivElement>();
  const colors = new Map(fish?.map((f) => [f.fish_id, f.color]));
  const camera = CAMERA_CHANNELS.map((c) => ({ channel: c, ...receipt.channels[c] }));
  const cameraBits = camera.reduce((n, c) => n + c.credited_bits, 0);
  const fishTotal = receipt.fish.reduce((n, f) => n + f.bits, 0);
  const describe = camera
    .map((c) => `${CHANNEL_META[c.channel].label} ${formatBits(c.credited_bits)}`)
    .join(', ');
  return (
    <div className="receipt">
      <div className="receipt__head">
        <span className="eyebrow">Entropy receipt</span>
        <span className="muted">
          {receipt.frames && (
            <>
              frames {receipt.frames[0].toLocaleString('en')}–{receipt.frames[1].toLocaleString('en')} ·{' '}
            </>
          )}
          {receipt.mode} · {formatClock(receipt.timestamp_ms)}
        </span>
      </div>
      {cameraBits > 0 ? (
        <div
          className="segbar"
          ref={barRef}
          role="img"
          aria-label={`Camera entropy in this key: ${describe}`}
        >
          {camera
            .filter((c) => c.credited_bits > 0)
            .map((c) => {
              const share = c.credited_bits / cameraBits;
              return (
                <div
                  key={c.channel}
                  className="segbar__part"
                  style={{ flexGrow: share, background: CHANNEL_META[c.channel].color }}
                  title={`${CHANNEL_META[c.channel].label}: ${formatBits(c.credited_bits)} (${Math.round(share * 100)}%)`}
                >
                  {share * barWidth >= LABEL_MIN_PX && (
                    <span className="segbar__label">{Math.round(share * 100)}%</span>
                  )}
                </div>
              );
            })}
        </div>
      ) : (
        <p className="muted">No camera credit in this draw: the key is exactly as strong as os.urandom.</p>
      )}
      <dl className="receipt__channels">
        {camera.map((c) => (
          <div key={c.channel}>
            <dt>
              <span
                className="legend-swatch"
                style={{ background: CHANNEL_META[c.channel].color }}
                aria-hidden
              />
              {CHANNEL_META[c.channel].label}
            </dt>
            <dd>
              {formatBits(c.credited_bits)}
              <StatusBadge look={HEALTH_LOOK[c.health]} size="sm" />
            </dd>
          </div>
        ))}
        <div>
          <dt>
            <span className="legend-swatch" style={{ background: CHANNEL_META.system.color }} aria-hidden />
            System, os.urandom
          </dt>
          <dd>+{formatBits(receipt.channels.system.credited_bits)}</dd>
        </div>
      </dl>
      {receipt.fish.length > 0 && (
        <div className="contrib">
          <div className="eyebrow">Fish that contributed</div>
          <ul className="contrib__list">
            {receipt.fish.map((f) => (
              <li key={f.fish_id} className="contrib__row">
                <span
                  className="swatch"
                  style={{ background: colors.get(f.fish_id) ?? 'var(--neutral)' }}
                  aria-hidden
                />
                <span className="contrib__name">{f.name ?? f.fish_id}</span>
                <span className="contrib__track">
                  <span
                    className="contrib__bar"
                    style={{ width: `${fishTotal > 0 ? (f.bits / fishTotal) * 100 : 0}%` }}
                  />
                </span>
                <span className="contrib__bits">{formatBits(f.bits)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="receipt__foot">
        <StatusBadge
          look={LIVENESS_LOOK[receipt.liveness.state]}
          size="sm"
          label={`Liveness ${LIVENESS_LOOK[receipt.liveness.state].label.toLowerCase()}`}
        />
        <span className="muted">
          {receipt.liveness.confidence_bits.toFixed(1)} bits of LED confidence at draw time
        </span>
      </div>
    </div>
  );
}

/**
 * Encrypt demo (DESIGN 3.3): the server draws a fresh key from PufferRNG,
 * seals the text with XChaCha20-Poly1305, and - because this is a demo -
 * hands the key to the browser so the round trip and the tamper test can be
 * shown. Real keys never leave the machine that generates them.
 */
export function EncryptPanel({ fish }: { fish: FishCard[] | undefined }) {
  const [plaintext, setPlaintext] = useState('Meet me by the coral reef at noon.');
  const [sealed, setSealed] = useState<EncryptResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [decryption, setDecryption] = useState<Decryption>({ kind: 'none' });
  const [tampered, setTampered] = useState<Tampered | null>(null);
  const size = utf8(plaintext).length;

  const encrypt = async () => {
    setBusy(true);
    setError(null);
    setDecryption({ kind: 'none' });
    setTampered(null);
    try {
      setSealed(await api.encrypt(plaintext));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const decrypt = async (ciphertext: string, isTampered: boolean) => {
    if (!sealed) return;
    setDecryption({ kind: 'pending', tampered: isTampered });
    try {
      const result = await api.decrypt(ciphertext, sealed.key);
      setDecryption({ kind: 'ok', plaintext: result.plaintext });
    } catch (e) {
      setDecryption({
        kind: 'failed',
        detail: e instanceof Error ? e.message : String(e),
        tampered: isTampered,
      });
    }
  };

  const tamper = () => {
    if (!sealed) return;
    const t = tamperOneByte(sealed.ciphertext);
    if (!t) return;
    setTampered(t);
    void decrypt(t.ciphertext, true);
  };

  return (
    <Panel
      className="area-encrypt"
      title={
        <>
          Encrypt with fish <span className="demo-tag">Demo</span>
        </>
      }
      subtitle="XChaCha20-Poly1305 with a fresh 32-byte key from the fish-seeded RNG."
    >
      <div className="demo-warning" role="note">
        <Icon name="alert" size={14} />
        <span>
          Demo only: the key is sent to your browser so you can see the round trip. Real keys never leave the
          machine that generates them.
        </span>
      </div>
      <form
        className="encrypt-form"
        onSubmit={(e) => {
          e.preventDefault();
          void encrypt();
        }}
      >
        <label className="field">
          <span className="field__label">Message</span>
          <textarea
            className="field__input"
            value={plaintext}
            rows={3}
            maxLength={MAX_PLAINTEXT_BYTES}
            onChange={(e) => setPlaintext(e.target.value)}
          />
          <span className={`field__hint${size > MAX_PLAINTEXT_BYTES ? ' field__hint--error' : ''}`}>
            {size.toLocaleString('en')} / {MAX_PLAINTEXT_BYTES.toLocaleString('en')} bytes
          </span>
        </label>
        <button
          type="submit"
          className="button button--primary"
          disabled={busy || size === 0 || size > MAX_PLAINTEXT_BYTES}
        >
          <Icon name="lock" size={15} />
          {busy ? 'Encrypting…' : 'Encrypt'}
        </button>
      </form>
      {!sealed && (
        <ol className="recipe" aria-label="How the key is made">
          <li>
            <span>
              <strong>Draw</strong> the pool: BLAKE2b digests of everything the camera channels absorbed, plus
              the credit ledger saying which fish earned what.
            </span>
          </li>
          <li>
            <span>
              <strong>Seed</strong>{' '}
              <span className="mono">BLAKE2b-256(key = os.urandom(32), purpose ‖ counter ‖ pool)</span>
            </span>
          </li>
          <li>
            <span>
              <strong>Expand</strong> with libsodium&apos;s ChaCha20 generator into a 32-byte key and a
              24-byte nonce.
            </span>
          </li>
          <li>
            <span>
              <strong>Seal</strong> with XChaCha20-Poly1305; the receipt shows where the entropy came from.
            </span>
          </li>
        </ol>
      )}
      {error && (
        <p className="inline-error" role="alert">
          <Icon name="x" size={14} /> {error}
        </p>
      )}
      {sealed && (
        <div className="sealed">
          <dl className="sealed__fields">
            <div>
              <dt>Key id</dt>
              <dd className="mono">{sealed.key_id}</dd>
            </div>
            <div>
              <dt>
                Key <span className="muted">(demo only)</span>
              </dt>
              <dd className="mono mono--wrap">
                <span>{sealed.key}</span>
                <CopyButton text={sealed.key} label="Copy key" />
              </dd>
            </div>
            <div>
              <dt>Ciphertext · base64url(nonce ‖ ciphertext ‖ tag)</dt>
              <dd className="mono mono--wrap mono--block">
                <span>
                  {tampered ? (
                    <DiffText original={sealed.ciphertext} changed={tampered.ciphertext} />
                  ) : (
                    sealed.ciphertext
                  )}
                </span>
                <CopyButton text={tampered?.ciphertext ?? sealed.ciphertext} label="Copy ciphertext" />
              </dd>
            </div>
          </dl>
          <div className="button-row">
            <button
              type="button"
              className="button"
              onClick={() => {
                setTampered(null);
                void decrypt(sealed.ciphertext, false);
              }}
              disabled={decryption.kind === 'pending'}
            >
              <Icon name="unlock" size={15} />
              Decrypt
            </button>
            <button
              type="button"
              className="button button--danger"
              onClick={tamper}
              disabled={decryption.kind === 'pending'}
            >
              <Icon name="zap" size={15} />
              Tamper one byte
            </button>
          </div>
          {decryption.kind === 'ok' && (
            <div className="outcome outcome--good" role="status">
              <Icon name="check" size={16} />
              <div>
                <strong>Authenticated and decrypted</strong>
                <p className="outcome__text">{decryption.plaintext}</p>
              </div>
            </div>
          )}
          {decryption.kind === 'failed' && (
            <div className="outcome outcome--critical" role="alert">
              <Icon name="x" size={16} />
              <div>
                <strong>{decryption.tampered ? 'Tampering detected' : 'Decryption failed'}</strong>
                <p className="outcome__text">
                  {decryption.tampered && tampered && (
                    <>
                      Flipped one bit of byte {tampered.offset}. Poly1305 rejected the whole message, so no
                      plaintext was released.{' '}
                    </>
                  )}
                  <span className="muted">Server: {decryption.detail}</span>
                </p>
              </div>
            </div>
          )}
          <Receipt receipt={sealed.receipt} fish={fish} />
        </div>
      )}
    </Panel>
  );
}
