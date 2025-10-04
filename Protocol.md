# How Quantum Algorithms Threaten Blockchains (Plain, Technical Explanation)

## Shor’s Algorithm (Impact on Asymmetric Crypto)

- **What it breaks:** Integer factoring and discrete logarithm problems used by RSA, DSA, ECDSA, ECDH, etc.
- **Blockchain impact:** Most blockchains rely on ECDSA/ECDH (or similar) for address keys, transaction signatures, and key exchange. A large enough quantum computer running Shor can recover private keys from public keys or signatures quickly. That would let an adversary:
    - Forge signatures and spend funds or submit arbitrary transactions as any account.
    - Derive ephemeral ECDH secrets and recover session keys used for off‑chain authenticated exchange.
- **Real risk:** Harvest‑Now, Decrypt‑Later (HNDL) — adversaries can record ciphertexts and authenticated exchanges today and decrypt later once quantum capability exists, exposing long‑lived data.

## Grover’s Algorithm (Impact on Symmetric Crypto and Hash Preimage Search)

- **What it does:** Gives a quadratic speed-up for unstructured search problems.
- **Hash functions and symmetric keys:** Grover reduces the security of an $n$‑bit symmetric key or $n$‑bit preimage search from $\sim2^n$ classical work to $\sim2^{n/2}$ quantum work.
- **Blockchain impact:**
    - Proof‑of‑Work (PoW) systems that rely on hash preimage difficulty will see effective difficulty halved in exponent (i.e., $2^{256} \rightarrow 2^{128}$ work for a 256-bit hash) for a full quantum attack — still very large, but the safety margin is reduced.
    - For hash-based commitments, signatures based on hash chains (if used) or puzzles lose some security margin.
- **Mitigation:** Double the symmetric/key/hash length (e.g., move from 128-bit symmetric to 256-bit symmetric equivalent security), or adopt quantum‑resistant symmetric and hash parameter sizes.

**Summary:** Shor is the existential threat for public-key signatures and key exchange used in blockchains; Grover reduces symmetric/hash security (quadratic speed-up). For PQBFL, the most urgent problems are signature/key-exchange security for both on-chain identities and off‑chain authenticated model transfer.

# High-level Migration Strategy for PQBFL (Design + Rationale)

**Goal:** Make PQBFL resistant to quantum attacks while keeping the hybrid, ratcheting, and on‑chain/off‑chain architecture.

**Key principles:**
- Use hybrid constructions (classical + post‑quantum) during transition: combine classical algorithms (ECDH/ECDSA) with PQC primitives (KEMs, signatures) so security holds if at least one primitive remains secure.
- Prefer KEMs and signature schemes that have reached NIST standardization or strong community review (Kyber / CRYSTALS‑Kyber for KEM, Dilithium/XMSS/etc for signatures).
- Keep heavy payloads off‑chain; store small PQ metadata on‑chain (hashes of public keys or PQ public keys), to limit gas and transaction cost.
- Upgrade ratchet design to support post‑quantum ratcheting: incorporate PQ KEM outputs into root key derivation and gracefully rotate to PQ signatures once available.

## Step-by-step Plan (Concrete)

1. **Choose post‑quantum primitives**
    - **KEM (key-encapsulation / key-exchange):**
        - Primary: CRYSTALS‑Kyber (Kyber) — NIST PQC selected KEM (Kyber768 recommended for 128-bit security level).
        - Alternative/backup: NTRU / Saber family depending on libraries available.
    - **Signatures:**
        - CRYSTALS‑Dilithium (NIST PQC) is a good primary choice.
        - Current blockchains (Ethereum) expect ECDSA/secp256k1 signatures — migrating on-chain to PQ signatures is nontrivial. Use hybrid signatures off‑chain and record classical blockchain signatures for on‑chain verification where required.
    - **Symmetric primitives / hashes:**
        - Use SHA‑256 / SHA‑384 for hashing and HKDF; pick 256/384/512 bit sizes consistent with mitigation for Grover.
        - Use AES‑256 or an AEAD like AES‑GCM or ChaCha20‑Poly1305 with 256-bit keys.

2. **Hybrid session establishment**
    - ECDH + Kyber KEM combined via KDF to produce root key.
    - All KEM operations use Kyber (or Kyber + another PQ KEM for multi-KEM hybrid) and keep ECDH as an additional layer if desired.
    - **Derivation example:**
        - Let $S_{kem}$ = Kyber shared secret (encapsulation/decapsulation).
        - Let $S_{ecdh}$ = ECDH shared secret.
        - $\text{RootKey} = \text{HKDF-Extract}(\text{SALT}, S_{kem} || S_{ecdh} || \text{context})$

3. **Post‑quantum signatures and authentication**
    - **Off‑chain messaging:** Move to PQ signatures (Dilithium) where possible for signing the encrypted model blobs and key exchange messages.
    - **On‑chain interactions:**
        - Continue to use existing on‑chain signature scheme to interact with existing blockchains (ECDSA). Instead of publishing full PQ public keys on-chain (large), store a compact commitment (hash) of the PQ public key and include an ECDSA-signed anchor that links your blockchain identity to PQ pubkey hash.
    - **Hybrid signatures:** When verifying an off‑chain message, accept a message as valid if either classical or PQ signature verifies, or require both for high security.

4. **Ratchet adaptation**
    - Asymmetric ratchet: use KEM (Kyber) + XDH/ECDH concept but ensure KEM replaces or augments the classical algorithm.
    - Symmetric ratchet: derive chain/model keys using HKDF-SHA384.
    - Root rotation: when server rotates KEM keys, clients perform Kyber encapsulation to get new $S_{kem}$ and new root key.
    - Ensure the ratchet design discards older keys securely and that root key derivation includes unique context.

5. **On-chain data size and gas considerations**
    - PQ public keys and PQ signatures are larger than classical ones. Don't publish full PQ keys on-chain unless absolutely required.
    - Store compact hashes (e.g., SHA-256 of PQ public keys) on-chain and use IPFS or off‑chain servers to publish full PQ keys and certificates.
    - For accountability, record PQ key hash + ECDSA anchor signature on-chain.

6. **Testing and transition strategy**
    - Hybrid-first deployment: use both PQ and classical simultaneously and require both for critical operations.
    - Maintain backward compatibility: clients without PQ capability can still participate (but flagged as legacy) during transition.
    - Build monitoring to check for PQ adoption and schedule a deprecation of classical-only participants.

# Concrete Implementation Guidance and Examples

## Libraries and Ecosystem

- **Python:**
    - pqcrypto or pqcrypto‑wrappers (Kyber)
    - liboqs + python bindings: Open Quantum Safe's liboqs (via python bindings)
    - PyCryptodome for symmetric crypto (AES, HMAC) and HKDF
    - web3.py for interacting with Ethereum-style blockchains
- **Go/JS:** liboqs has bindings for multiple languages; JS implementations exist for some PQ algorithms.
- **Smart contracts:** Solidity (for Ethereum): storing hashes and minimal metadata only.

## Example Message / Key Derivation (Pseudo‑Python)

