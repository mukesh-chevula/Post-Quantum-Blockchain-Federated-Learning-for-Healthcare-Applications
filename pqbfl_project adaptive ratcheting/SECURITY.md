# Security Hardening — Side-Channel Resistant PQBFL

This document describes all side-channel attack mitigations applied to the
**pqbfl_project sidechannel resistant** variant compared to the original
`pqbfl_project`.

---

## 1. Constant-Time Comparisons

**Vulnerability:** Python's `==` operator on bytes/strings short-circuits on the
first mismatched byte, allowing a timing oracle that can leak secret values one
byte at a time.

**Mitigation:** All security-critical comparisons now use `hmac.compare_digest()`
which runs in constant time regardless of how many bytes match.

| Location | What is compared |
|---|---|
| `utils.py: secure_compare()` | Generic bytes/string |
| `utils.py: secure_bytes_compare()` | Raw bytes (keys, hashes) |
| `utils.py: secure_hash_compare()` | SHA-256 hash vs expected hex |
| `crypto/ethsig.py: verify_signer()` | Recovered address vs expected |
| `protocol.py` | All hash, key, and address checks |
| `scripts/demo_end_to_end.py` | Root key and model key checks |

---

## 2. Authenticated Encryption (AES-256-GCM)

**Vulnerability:** The original `PQBFL/utils.py` used AES-CTR mode without
authentication — an attacker could flip ciphertext bits to modify plaintext model
weights without detection.

**Mitigation:** `crypto/aead.py` uses AES-256-GCM (AEAD) exclusively, providing
both confidentiality and integrity. GCM's authentication tag detects any
ciphertext tampering.

---

## 3. Random Nonces

**Vulnerability:** Deterministic nonces derived from round numbers leaked round
information and risked nonce reuse across sessions.

**Mitigation:** `crypto/aead.py` generates a fresh 12-byte random nonce via
`os.urandom(12)` for every encryption. The nonce is prepended to the ciphertext
so both parties can reconstruct it.

---

## 4. Constant-Time Kyber (pqcrypto C Backend)

**Vulnerability:** The `kyber-py` pure-Python library used in `Prototype/` has
data-dependent timing in NTT, polynomial multiplication, and decapsulation.

**Mitigation:** This project uses `pqcrypto` (liboqs C bindings) which provides
constant-time NTT, constant-time polynomial arithmetic, and a constant-time
comparison in the decapsulation FO transform. The toy fallback emits loud
runtime warnings so it cannot be silently deployed.

---

## 5. Signature Verification (Always Enabled)

**Vulnerability:** In `PQBFL/utils.py:verify_sign()`, the actual verification
code was commented out — signatures were parsed but **never checked**.

**Mitigation:** `crypto/ethsig.py:verify_signer()` always performs full ECDSA
signature recovery and verifies the recovered address matches the expected signer
using `hmac.compare_digest`.

---

## 6. Safe Serialization (No Pickle)

**Vulnerability:** `PQBFL/utils.py` and `aggregate.py` used `pickle.loads()` to
deserialize client-submitted model data — enabling Remote Code Execution.

**Mitigation:** `fl/model.py` serializes with `numpy.savez()` and deserializes
with `numpy.load(allow_pickle=False)`. This format cannot execute arbitrary code.
JSON is used for metadata via `utils.json_dumps_canonical()`.

---

## 7. Random KDF Salts

**Vulnerability:** Salts were zero-byte arrays incremented as simple counters.
An attacker who knows the round number knows the exact salt, reducing ratcheting
security.

**Mitigation:** `crypto/kdf.py` provides `generate_random_salt()` using
`os.urandom(32)`. The protocol coordinator should distribute random salts during
session establishment instead of using predictable counters.

---

## 8. Constant-Time Key Validation

**Vulnerability:** Key length checks raised exceptions for short keys but
silently truncated long keys — creating a timing difference.

**Mitigation:** `crypto/aead.py:_validate_key()` performs a single uniform check
that rejects any key that is not exactly 32 bytes, with no branching based on
key content.

---

## 9. Private Key Handling

**Vulnerability:** Private keys were passed via `sys.argv` CLI arguments, visible
in process lists and shell history.

**Mitigation:** The demo derives keys from an HD wallet mnemonic
(`derive_hardhat_account`). Production deployments should use environment
variables, hardware security modules, or OS keyring APIs.

---

## 10. Constant-Time ECDH and ECDSA

The `cryptography` library (pyca/cryptography) delegates to OpenSSL which
provides constant-time scalar multiplication for ECDH and constant-time ECDSA
signing. This is already the backend used in `crypto/ecdh.py`.

---

## 11. Memory Safety for Secrets

Python's memory management makes it difficult to guarantee secret zeroization.
For production deployments, consider:

- Using `mlock()` to prevent key material from being swapped to disk
- Wrapping critical operations in a C extension with explicit `memset` cleanup
- Using hardware security modules (HSMs) for key storage

---

## Summary of Changes by File

| File | Change |
|---|---|
| `crypto/aead.py` | Random nonces, strict key validation, nonce prepended to ciphertext |
| `crypto/ethsig.py` | Added `verify_signer()` with constant-time address comparison |
| `crypto/kdf.py` | Added `generate_random_salt()`, documented constant-time HMAC |
| `crypto/kyber.py` | Loud warnings for toy fallback, backend diagnostic function |
| `utils.py` | Added `secure_compare`, `secure_bytes_compare`, `secure_hash_compare` |
| `protocol.py` | All comparisons use constant-time functions |
| `fl/model.py` | `np.load(allow_pickle=False)`, input validation |
| `scripts/demo_end_to_end.py` | Uses `verify_signer`, `secure_bytes_compare` everywhere |
