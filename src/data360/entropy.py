"""Cryptographically secure randomness for retry/refresh jitter (CWE-331).

Veracode reports "Insufficient Entropy" wherever the ``random`` module's
Mersenne Twister is used, because its output is predictable and can be
reproduced from a seed. Nothing here protects secrets — the values are only
backoff jitter and load-test case selection — but the policy requires a trusted
cryptographic generator, so every value comes from :mod:`secrets` (the OS CSPRNG:
``os.urandom`` / ``getrandom``).

``secrets`` has no float API, so :func:`uniform_jitter` builds one from
``secrets.randbelow``. The result is uniform over a discrete grid rather than
continuous — irrelevant for jitter, where only decorrelation matters.
"""

from __future__ import annotations

import secrets

__all__ = ["uniform_jitter"]

# Resolution of the discrete uniform grid. 10k steps is far finer than any
# jitter needs while keeping randbelow cheap.
_STEPS = 10_000


def uniform_jitter(low: float, high: float) -> float:
    """Return a value in ``[low, high)`` drawn from the OS CSPRNG.

    ``high <= low`` returns ``low`` (no band to jitter within).
    """
    if high <= low:
        return low
    return low + (high - low) * (secrets.randbelow(_STEPS) / _STEPS)
