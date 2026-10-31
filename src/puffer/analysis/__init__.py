"""Statistical analysis of random bit streams: SP 800-22, ent, dieharder.

Everything here is *test tooling* for reports and the dashboard - it never
produces or touches secret material. Passing these tests is necessary but
never sufficient evidence of a good source (DESIGN.md rule 4): hashed or
conditioned output passes them whatever its entropy, so results must always
be presented together with what data produced them.

* :mod:`.bits` - byte/bit conversion (MSB first) and NIST STS file formats.
* :mod:`.sp800_22` - the 15 NIST SP 800-22 rev1a tests, the suite runner and
  the multi-sequence analysis of section 4.2.
* :mod:`.ent` - an in-process equivalent of John Walker's ``ent``.
* :mod:`.external` - wrappers for the real ``ent`` and ``dieharder`` binaries.
"""

from puffer.analysis.bits import (
    BitArray,
    as_bits,
    ascii_to_bits,
    bits_to_ascii,
    bits_to_bytes,
    bytes_to_bits,
    load_bits,
    save_bits,
    split_sequences,
)
from puffer.analysis.ent import EntStats, ent_stats, ent_stats_file
from puffer.analysis.external import (
    DieharderAssessment,
    DieharderResult,
    ExternalToolError,
    ToolNotFoundError,
    ToolTimeoutError,
    dieharder_available,
    ent_available,
    run_dieharder,
    run_ent,
)
from puffer.analysis.sp800_22 import (
    SUITE_TESTS,
    MultiSuiteResult,
    SubtestSummary,
    SuiteResult,
    TestResult,
    run_multi,
    run_suite,
)

__all__ = [
    "SUITE_TESTS",
    "BitArray",
    "DieharderAssessment",
    "DieharderResult",
    "EntStats",
    "ExternalToolError",
    "MultiSuiteResult",
    "SubtestSummary",
    "SuiteResult",
    "TestResult",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "as_bits",
    "ascii_to_bits",
    "bits_to_ascii",
    "bits_to_bytes",
    "bytes_to_bits",
    "dieharder_available",
    "ent_available",
    "ent_stats",
    "ent_stats_file",
    "load_bits",
    "run_dieharder",
    "run_ent",
    "run_multi",
    "run_suite",
    "save_bits",
    "split_sequences",
]
