# Security policy

puffer-cipher is a research and portfolio project. It has not been audited
by a third party. Read the [threat model](README.md#threat-model) before
relying on it for anything that matters.

## Design guarantees

- All cryptography comes from libsodium (via PyNaCl) and BLAKE2b from
  Python's standard library. The project implements protocols and formats,
  never primitives.
- Every secret output is keyed by fresh `os.urandom` bytes, so it is never
  weaker than the operating system's generator, even if the camera feed is
  fully controlled by an attacker.
- Beacon values are public. Never use them as keys.

## Reporting a vulnerability

Please report security issues privately through
[GitHub security advisories](https://github.com/lukamaga/puffer-cipher/security/advisories/new)
rather than in a public issue. Include the affected component, a minimal
reproduction and the impact you expect. You will get an acknowledgement
within a week.

Findings in scope include anything that lets an attacker predict or bias
secret output, forge or fork the beacon chain, pass the LED liveness check
with a non-live video, recover plaintext or keys from `.puf` files, or make
the server leak secrets.
