"""puffer-cipher: cryptographic randomness seeded by live fish.

The innovation lives in the entropy source and the engineering around it,
never in the cipher: all cryptography is delegated to libsodium (PyNaCl)
and Python's hashlib BLAKE2b.
"""

__version__ = "0.1.0"
