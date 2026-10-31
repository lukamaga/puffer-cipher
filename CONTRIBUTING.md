# Contributing

Thanks for taking an interest in puffer-cipher. Issues and pull requests are
welcome.

## Development setup

```bash
git clone https://github.com/lukamaga/puffer-cipher.git
cd puffer-cipher
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cd dashboard && npm ci && cd ..
```

Optional tools used by the statistical tests: `sudo apt install ent dieharder`.

## Checks

Every pull request runs the same checks as CI:

```bash
ruff check src tests scripts && ruff format --check src tests scripts
mypy
pytest -m "not slow"
cd dashboard && npm run lint && npm test && npm run build
```

`pytest -m slow` runs the long statistical suites (about two minutes).

## Ground rules

1. **No home-made cryptography.** Use libsodium through PyNaCl, or
   `hashlib.blake2b`. New formats must be specified in
   [docs/DESIGN.md](docs/DESIGN.md) first.
2. **Never weaker than the OS.** Any path that produces secret bytes must be
   keyed by fresh `os.urandom` output.
3. **Credit is earned.** Entropy is credited only from samples with a
   min-entropy estimate and only while health checks pass.
4. **Honest numbers.** Results must say which data produced them
   (simulator, recorded video or live camera).

Beacon wire format changes must regenerate the shared test vectors with
`python -m puffer.beacon.vectors --out tests/vectors/beacon_vectors.json`,
because the dashboard verifies pulses with the same file.
