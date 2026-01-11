# PQBFL 

This is a runnable project of **PQBFL: A Post‑Quantum Blockchain‑based Protocol for Federated Learning**.

What it includes:
- **On-chain (Hardhat / local Ethereum):** a Solidity smart contract implementing the paper’s project/client/task/update/feedback lifecycle.
- **Off-chain (Python):** session establishment + **hybrid keying** (Kyber KEM + ECDH) and **ratcheting** to derive per-round model keys.
- **Encrypted model exchange:** AES‑256‑GCM using per‑round model keys.
- **Federated learning loop:** a small FedAvg demo (logistic regression) over synthetic non‑IID client datasets.

## 1) Prereqs
- Node.js 18+ (for Hardhat)
- Python 3.9–3.12 recommended (the `pqcrypto` wheels used for ML‑KEM/Kyber are typically not available for Python 3.14)

## 2) Start the local blockchain
In terminal 1:

```bash
cd pqbfl_project/chain
npm install
npm run compile
npm run node
```

If you see a Hardhat warning about an unsupported Node.js version, switch to Node 18 or 20 (e.g., via `nvm`).

If you see `Error: listen EADDRINUSE: address already in use 127.0.0.1:8545`, it means something is already running on port 8545 (often another Hardhat node). Either:
- Stop the existing process (Ctrl+C in the terminal that’s running it), or
- Start Hardhat on a different port, e.g. `npx hardhat node --port 8546` and set `PQBFL_CHAIN_URL=http://127.0.0.1:8546`.

To see what is using the port on macOS:

```bash
lsof -nP -iTCP:8545 -sTCP:LISTEN
```

Keep this running.

## 3) Run the end-to-end demo
In terminal 2:

```bash
cd pqbfl_project/python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pqbfl.scripts.demo_end_to_end
```

## 4) Run the web UI
The UI lets you start/stop a local Hardhat node and run the demo with sliders, then plots accuracy per round.

```bash
cd pqbfl_project/python
source .venv/bin/activate
pip install -r requirements.txt
streamlit run ui_app.py
```

If your default `python3` is 3.14+, use a supported interpreter (macOS often has `/usr/bin/python3` as 3.9):

```bash
cd pqbfl_project/python
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pqbfl.scripts.demo_end_to_end
```

You should see:
- Contract deployment address
- On-chain registrations
- Per-round test accuracy logs

## Notes on mapping to the paper
- **Session establishment:** follows the paper’s “Session establishment” steps (server sends `kpk_b, epk_b`, client encapsulates to `ct`, both derive `RK_j`).
- **Model confidentiality & replay protection:** per-round model keys are derived via a symmetric ratchet, and the AEAD nonce/AAD include the round + direction.
- **Hybrid crypto:** Kyber512 (post‑quantum KEM) + ECDH over secp256k1.

## Where things are placed
- Smart contract: pqbfl_project/chain/contracts/PQBFL.sol
- Demo runner: pqbfl_project/python/pqbfl/scripts/demo_end_to_end.py
- Ratchet + crypto: pqbfl_project/python/pqbfl/protocol.py and pqbfl_project/python/pqbfl/crypto/
