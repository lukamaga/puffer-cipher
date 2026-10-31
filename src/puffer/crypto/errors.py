"""Exception hierarchy of the cryptography layer.

Callers (CLI, REST API) need to tell *why* decryption failed without ever
learning anything an attacker could exploit, so the hierarchy is shaped by
what a user can do about each failure:

* :class:`FormatError` - the bytes are not a well-formed puffer object
  (wrong magic or version, unknown KDF, out-of-range header field, garbage
  base64, trailing data). Nothing secret was touched.
* :class:`WrongKeyError` - the object names a different key (key-id
  mismatch) or a different kind of credential (key file vs password).
  Raised *before* any decryption is attempted, so it reveals nothing beyond
  the public key id already stored in the header.
* :class:`AuthenticationError` - libsodium rejected a Poly1305 tag: the
  ciphertext, header or tag was modified, chunks were reordered, or the
  password / key is wrong. libsodium deliberately does not say which, and
  neither do we.
* :class:`TruncatedError` - the encrypted stream ended before its final
  chunk. Every chunk released so far was authentic, but the message is
  incomplete and must not be treated as the whole plaintext.

:class:`FormatError` also derives from :class:`ValueError` (like
``json.JSONDecodeError`` and ``binascii.Error``), so generic parsing code
that already catches ``ValueError`` keeps working.
"""

from __future__ import annotations


class PufferCryptoError(Exception):
    """Base class: every failure to parse, authenticate or decrypt puffer data."""


class FormatError(PufferCryptoError, ValueError):
    """Malformed input: bad magic, version, KDF, header field, encoding or trailing data."""


class WrongKeyError(PufferCryptoError):
    """The supplied key or credential type does not match the one the data names."""


class AuthenticationError(PufferCryptoError):
    """Authentication tag verification failed (tampering, reordering or wrong secret)."""


class TruncatedError(PufferCryptoError):
    """The encrypted stream ended before the chunk carrying ``TAG_FINAL``."""


class InsecurePermissionsWarning(UserWarning):
    """A secret key file is readable or writable by users other than its owner.

    Emitted (not raised) on load, in the spirit of OpenSSH's permission
    check: refusing outright would break restores from backups and copies
    across filesystems, but silently accepting a world-readable key would
    hide a real exposure.
    """
