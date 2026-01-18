# PQBFL — Post-Quantum Blockchain-based Protocol for Federated Learning


## Overview

PQBFL is a hybrid, post-quantum-aware protocol that secures federated learning (FL) by combining post-quantum cryptography (PQC), traditional cryptographic primitives, and blockchain. The design targets threats introduced by future quantum computers (including Harvest-Now, Decrypt-Later attacks) while addressing classical FL security and privacy problems such as model interception, tampering, membership/source inference, replay attacks, free-riding, and single-point failures.

Key ideas:

- Use a hybrid cryptographic approach (classical ECDH + post-quantum KEM Kyber) to derive root keys and obtain resilience if either primitive remains secure.
- Employ symmetric and asymmetric ratcheting to provide forward secrecy and post-compromise security across multiple FL rounds without exchanging heavy public keys each round.
- Use blockchain as a decentralized ledger and key-establishment facilitator (store hashes of public keys, project/task lifecycle events, and reputation scores) while keeping bulk model traffic off-chain to reduce cost.

## Main contributions (condensed)

- A hybrid PQC + classical crypto protocol tailored to the iterative nature of FL that resists HNDL attacks.
- Ratcheting (asymmetric + symmetric) for per-round confidentiality, forward secrecy, and post-compromise recovery.
- A hybrid communication model (on-chain for metadata and accountability, off-chain for encrypted model transfers) to balance privacy, cost, and auditability.
- A blockchain-based reputation and transaction scheme for participant registration, task publication, model updates, and feedback.

## Protocol components

- Blockchain: smart contract to register projects/clients, publish tasks, record model update hashes and feedback, and manage reputation and lifecycle events.
- Server (aggregator): deploys project contract, publishes tasks, derives and rotates keys, validates and aggregates encrypted local models.
- Participant (client): registers to project, derives shared keys, encrypts and signs model updates, and submits model hashes on-chain while sending encrypted models off-chain.

## Cryptography & Key Management

- Signatures: ECDSA (used for blockchain transactions and off-chain message authentication).
- KEM: Kyber (post-quantum) for encapsulation/decapsulation in the asymmetric ratchet.
- ECDH: classical ECDH (e.g., NIST P-256) combined with KEM output to derive root keys.
- KDFs: HKDF-based extract-then-expand for symmetric key derivation and ratcheting.
- Symmetric encryption: AES-256 (or an equivalent AEAD) for encrypting model payloads.

Ratcheting

- Asymmetric ratchet: new KEM + ECDH public keys periodically rotate the root key (triggered by server) and provide post-compromise security.
- Symmetric ratchet: per-round chain/model keys derived from the root key using HKDF; old keys are discarded to provide forward secrecy.

## Security properties

PQBFL aims to meet the following guarantees:

- Authentication: mutual authentication via signatures and blockchain-anchored public key hashes.
- Confidentiality: per-round model encryption with ratcheting-derived keys; hybrid KEM+ECDH root key resists HNDL.
- Forward secrecy: symmetric ratchets ensure past model keys remain secure if a later key is compromised.
- Post-compromise security: asymmetric ratcheting refreshes root keys and recovers security after compromises.
- Replay protection: unique timestamps and per-round keys prevent replayed model updates from being accepted.
- Privacy & traceability: blockchain pseudonymity balances identity privacy with accountability via reputation scores.

## Performance (summary)

The paper provides an analysis of computation and communication costs. Highlights:

- Using a larger symmetric ratcheting range (more symmetric keys per asymmetric ratchet) reduces the need for expensive public-key ops and lowers total computation and communications overhead.
- Hybrid on-chain/off-chain architecture reduces blockchain data and gas costs by storing only hashes and metadata on-chain while model payloads are exchanged off-chain encrypted.
- Experimental setup in the paper used Kyber-768, ECDH (P-256), HKDF-SHA384, ECDSA-secp256k1, and AES-256. The authors found an efficient trade-off at roughly 10 symmetric ratchets per asymmetric ratchet for many workloads.

## Smart contract interface (high level)

The PQBFL smart contract (described in the paper) exposes events and functions such as:

- Events: RegClient, RegProject, Task (publish), Update, Feedback, ProjectTerminate
- Functions: RegisterProject, RegisterClient, PublishTask, UpdateModel, FeedbackModel, UpdateScore, FinishProject

This contract stores project metadata, participant registration hashes (e.g., hashed public keys), task deadlines, and update/feedback records.

## How to use (notes & implementation pointers)

This repository includes a small reference implementation under:

- `pqbfl_project/` (baseline implementation)
- `pqbfl_project new/` (enhanced implementation)

The original paper references an implementation on GitHub (HIGHer, 2024). Independently of that, for a working implementation you would typically:

1. Deploy the smart contract to a testnet or private chain. Ensure the contract records hashed public keys and event logs for clients to watch.
2. Have the server generate KEM (Kyber) and ECDH key pairs and publish hashes of their public keys in the RegisterProject transaction.
3. Clients generate ECDH keys and register using the RegisterClient transaction (publish hash of client ECDH pk).
4. Use off-chain channels for signed, encrypted message exchange (containing public keys, ct, encrypted model blobs). Store model hashes and metadata on-chain.
5. Implement symmetric and asymmetric ratcheting according to the paper: derive chain/model keys via HKDF and rotate root keys after L_j symmetric steps.

Suggested libraries and tools

- Python: pqcrypto (Kyber), PyCryptodome for AES/HMAC, web3.py for smart-contract interaction.
- Smart contracts: Solidity + Hardhat or Truffle for development and gas estimation.

## Improvements in `pqbfl_project new/` vs `pqbfl_project/`

The `pqbfl_project new/` folder is an iteration over `pqbfl_project/` with the following practical improvements:

- **Modernized crypto primitives for the off-chain channel**
	- **AEAD:** switched from **AES-256-GCM** to **ChaCha20-Poly1305**.
	- **ECDH:** added **X25519 (Curve25519)** support and migrated session establishment to X25519.
	- **Signatures (off-chain):** added **Ed25519** signing/verification and migrated the demo protocol away from Ethereum-style ECDSA address recovery for off-chain messages.

- **Hashing / KDF upgrades (with safe fallback)**
	- Introduced a protocol-wide `hash32()` (prefers **BLAKE3-256**, falls back to **SHA-256**).
	- Updated HKDF extract/expand and the symmetric ratchet PRF to prefer **HMAC-(BLAKE3-256)** (fallback to HMAC-SHA256).
	- Added `blake3` to Python dependencies.

- **More realistic FL simulation knobs**
	- Added **partial participation** (client participation rate) and a simple **label-flip poisoning** toggle.
	- Exposed dataset/model seeds and training hyperparameters (LR, epochs, batch size, L2).

- **More robust aggregation options**
	- Added **coordinate-wise median** and **trimmed mean** as alternatives to FedAvg.
	- Added optional **L2 regularization** in the logistic-regression SGD trainer.

- **Streamlit UI enhancements**
	- UI now exposes the above simulation/training/aggregation settings so experiments can be configured without editing code.

## Limitations and future work

- The paper uses a hybrid classical + PQC approach; full migration to post-quantum signatures and post-quantum blockchains is left for future adoption as standards mature.
- Model-level privacy (e.g., defense against membership inference) is acknowledged as out-of-scope; future work suggests applying post-quantum homomorphic encryption or secure aggregation that is quantum-resistant.

## References

See the paper for a complete set of references. Key references include:

- Crystals-Kyber (NIST PQC)
- HKDF (Krawczyk)
- PQXDH / PQ3 (post-quantum messaging designs)
- Relevant FL + blockchain works cited in the paper (DAFL, BSAFL, LaF, BFL, BESIFL)

## Contact & citation

If you use PQBFL in research or an implementation, please cite:

Gharavi, H., Granjal, J., & Monteiro, E. "PQBFL: A Post-Quantum Blockchain-based Protocol for Federated Learning" (arXiv:2502.14464v1, 2025).

---

This is an initial README draft derived from the paper. I can expand sections with diagrams, the smart-contract ABI, example message flows, or a minimal demo (contract + client/server scripts). Tell me which you prefer next.
