"""Cryptography layer: formats and protocols on top of libsodium, never primitives.

Everything cryptographic here is a call into libsodium through PyNaCl
(XChaCha20-Poly1305, secretstream, Argon2id, Ed25519) or into
``hashlib.blake2b``. What this package designs is the *engineering around*
those primitives - byte formats, key files, failure semantics - specified
in DESIGN.md section 3:

* :mod:`~puffer.crypto.aead` - short messages, ``base64url(nonce || ct)``.
* :mod:`~puffer.crypto.fileformat` - the chunked, header-authenticated
  ``.puf`` file format with truncation detection and atomic output.
* :mod:`~puffer.crypto.keys` - key ids and ``0600`` JSON key files that
  carry the entropy receipt of the fish that made them.
* :mod:`~puffer.crypto.signing` - the beacon's persistent Ed25519 key.

Randomness is always injectable (``random_bytes=``) so the engine can feed
the fish-seeded ``PufferRNG``; the defaults use libsodium's CSPRNG.
"""

from puffer.crypto.aead import (
    MESSAGE_AAD,
    NONCE_BYTES,
    RandomBytes,
    decrypt_message,
    decrypt_message_b64,
    encrypt_message,
    encrypt_message_b64,
)
from puffer.crypto.encoding import b64url_decode, b64url_encode
from puffer.crypto.errors import (
    AuthenticationError,
    FormatError,
    InsecurePermissionsWarning,
    PufferCryptoError,
    TruncatedError,
    WrongKeyError,
)
from puffer.crypto.fileformat import (
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    Argon2Params,
    FileHeader,
    Kdf,
    decrypt_file,
    decrypt_stream,
    encrypt_file,
    encrypt_stream,
    read_header,
)
from puffer.crypto.keys import KEY_BYTES, KeyFile, key_id, load_key, save_key
from puffer.crypto.signing import load_or_create_signing_key, public_key_hex

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "KEY_BYTES",
    "MAX_CHUNK_SIZE",
    "MESSAGE_AAD",
    "MIN_CHUNK_SIZE",
    "NONCE_BYTES",
    "Argon2Params",
    "AuthenticationError",
    "FileHeader",
    "FormatError",
    "InsecurePermissionsWarning",
    "Kdf",
    "KeyFile",
    "PufferCryptoError",
    "RandomBytes",
    "TruncatedError",
    "WrongKeyError",
    "b64url_decode",
    "b64url_encode",
    "decrypt_file",
    "decrypt_message",
    "decrypt_message_b64",
    "decrypt_stream",
    "encrypt_file",
    "encrypt_message",
    "encrypt_message_b64",
    "encrypt_stream",
    "key_id",
    "load_key",
    "load_or_create_signing_key",
    "public_key_hex",
    "read_header",
    "save_key",
]
