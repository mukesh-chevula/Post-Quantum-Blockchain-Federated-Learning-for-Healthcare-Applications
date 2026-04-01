# PQBFL — Side-Channel Resistant Variant

This is a hardened fork of `pqbfl_project` with comprehensive side-channel
attack mitigations applied across the cryptographic, protocol, and data
handling layers.

## What Changed

| Category | Original Risk | Mitigation |
|---|---|---|
| Hash/key comparisons | `==` operator (timing oracle) | `hmac.compare_digest` everywhere |
| AEAD nonces | Deterministic from round number | Random `os.urandom(12)` per encryption |
| Signature verification | Disabled / non-constant-time | Always-on, constant-time address comparison |
| KEM backend | Pure-Python `kyber-py` (timing leaks) | `pqcrypto` C backend (constant-time) |
| Model serialization | `pickle.loads` (RCE risk) | `numpy.load(allow_pickle=False)` |
| KDF salts | Zero bytes / counters | `os.urandom(32)` random salts |
| Key management | CLI arguments (`sys.argv`) | HD wallet derivation / env vars |
| AES mode (PQBFL/) | CTR without MAC | GCM (authenticated) |

See [SECURITY.md](SECURITY.md) for detailed analysis of each mitigation.

## Quick Start

```bash
# Install dependencies
cd python
pip install -r requirements.txt

# Run tests (no blockchain needed)
python test_side_channel_hardening.py

# Run full demo (requires Hardhat node)
cd ../chain && npm install && npm run node &
cd ../python && python -m pqbfl.scripts.demo_end_to_end
```

## Project Structure

```
pqbfl_project sidechannel resistant/
├── SECURITY.md              ← all mitigations documented
├── README.md                ← this file
├── chain/                   ← Solidity smart contract (Hardhat)
├── docker/                  ← Docker configuration
└── python/
    ├── requirements.txt
    ├── test_side_channel_hardening.py   ← verification tests
    └── pqbfl/
        ├── crypto/          ← hardened cryptographic modules
        ├── chain/           ← blockchain contract client
        ├── fl/              ← federated learning (safe serialization)
        ├── scripts/         ← hardened end-to-end demo
        ├── protocol.py      ← hardened PQBFL protocol
        └── utils.py         ← constant-time comparison helpers
```
