"""Entropy core: channels, health tests, pool, RNG, receipts.

Data flows one way (DESIGN.md sections 1 and 4)::

    Frame / TrackingResult
        -> SensorChannel / FishChannel        (ChannelOutput: absorb + samples)
        -> HealthMonitor                      (SP 800-90B RCT/APT + domain checks)
        -> RollingEstimate + compute_credit   (credit from samples only)
        -> EntropyPool.absorb                 (BLAKE2b accumulators + ledger)
        -> PufferRNG.generate                 (keyed by os.urandom -> bytes + receipt)

The single most important property - output is never weaker than
``os.urandom`` - lives in :mod:`puffer.entropy.rng`. Everything else decides
how much *credit* the fish and the sensor honestly deserve.
:mod:`puffer.entropy.conditioning` is an unkeyed analysis-only path and must
never be used for secrets.
"""

from puffer.entropy.channels import (
    DEFAULT_H_CAP,
    SAFETY_FACTOR,
    FishChannel,
    RollingEstimate,
    SensorChannel,
    SensorChannelConfig,
    compute_credit,
)
from puffer.entropy.conditioning import ConditionedStream
from puffer.entropy.health import (
    AdaptiveProportionTest,
    ChannelHealth,
    CheckResult,
    FishPresenceCheck,
    FrozenFrameCheck,
    HealthConfig,
    HealthMonitor,
    LivenessCheck,
    LowEntropyCheck,
    RepetitionCountTest,
    SourceTimeoutCheck,
    apt_cutoff,
    rct_cutoff,
)
from puffer.entropy.pool import EntropyPool, PoolDraw
from puffer.entropy.receipt import (
    ChannelCredit,
    EntropyReceipt,
    FishCredit,
    ReceiptContext,
)
from puffer.entropy.rng import PufferRNG, derive_seed

__all__ = [
    "DEFAULT_H_CAP",
    "SAFETY_FACTOR",
    "AdaptiveProportionTest",
    "ChannelCredit",
    "ChannelHealth",
    "CheckResult",
    "ConditionedStream",
    "EntropyPool",
    "EntropyReceipt",
    "FishChannel",
    "FishCredit",
    "FishPresenceCheck",
    "FrozenFrameCheck",
    "HealthConfig",
    "HealthMonitor",
    "LivenessCheck",
    "LowEntropyCheck",
    "PoolDraw",
    "PufferRNG",
    "ReceiptContext",
    "RepetitionCountTest",
    "RollingEstimate",
    "SensorChannel",
    "SensorChannelConfig",
    "SourceTimeoutCheck",
    "apt_cutoff",
    "compute_credit",
    "derive_seed",
    "rct_cutoff",
]
