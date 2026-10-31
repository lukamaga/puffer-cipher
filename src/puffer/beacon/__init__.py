"""Public randomness beacon powered by fish (DESIGN.md 3.4).

Like the NIST Randomness Beacon 2.0 or drand, the beacon publishes one
signed pulse per period. Each pulse reveals a fresh 64-byte value drawn from
PufferRNG, precommits to the next one, links to the previous pulse's output
and carries a public summary of the entropy behind it. Anyone can verify the
chain offline with only the Ed25519 public key - in Python with
:func:`verify_chain`, in the browser with the dashboard's verifier, both
tested against the same vectors (:mod:`puffer.beacon.vectors`).

Beacon values are public: they must never be used as secret keys.

Modules: :mod:`~puffer.beacon.pulse` (format, signing, verification),
:mod:`~puffer.beacon.chain` (sequences and files),
:mod:`~puffer.beacon.store` (SQLite), :mod:`~puffer.beacon.service`
(scheduler), :mod:`~puffer.beacon.summary` (receipt to summary).
:mod:`puffer.beacon.vectors` is deliberately not imported here so that
``python -m puffer.beacon.vectors`` runs it exactly once.
"""

from puffer.beacon.chain import ChainError, ChainReport, verify_chain, verify_file
from puffer.beacon.pulse import (
    DOMAIN,
    MESSAGE_BYTES,
    VERSION,
    BeaconFormatError,
    ErrorCode,
    Pulse,
    VerificationResult,
    build_message,
    canonical_summary,
    create_pulse,
    verify_pulse,
)
from puffer.beacon.service import (
    BeaconEntropyError,
    BeaconService,
    BeaconStateError,
    BeaconTimingError,
    EntropySource,
)
from puffer.beacon.store import (
    BeaconStore,
    BeaconStoreError,
    ChainLinkError,
    PendingMismatchError,
    PendingReveal,
)
from puffer.beacon.summary import summary_from_receipt

__all__ = [
    "DOMAIN",
    "MESSAGE_BYTES",
    "VERSION",
    "BeaconEntropyError",
    "BeaconFormatError",
    "BeaconService",
    "BeaconStateError",
    "BeaconStore",
    "BeaconStoreError",
    "BeaconTimingError",
    "ChainError",
    "ChainLinkError",
    "ChainReport",
    "EntropySource",
    "ErrorCode",
    "PendingMismatchError",
    "PendingReveal",
    "Pulse",
    "VerificationResult",
    "build_message",
    "canonical_summary",
    "create_pulse",
    "summary_from_receipt",
    "verify_chain",
    "verify_file",
    "verify_pulse",
]
