from __future__ import annotations

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from pqbfl.utils import hash32


def nonce_for_round(round_num: int, label: str) -> bytes:
    if round_num < 0:
        raise ValueError("round_num must be >= 0")
    seed = f"pqbfl:{label}:{round_num}".encode("utf-8")
    return hash32(seed)[:12]


def aead_encrypt(key32: bytes, plaintext: bytes, *, aad: bytes, nonce: bytes) -> bytes:
    if len(key32) != 32:
        raise ValueError("ChaCha20-Poly1305 key must be 32 bytes")
    aead = ChaCha20Poly1305(key32)
    return aead.encrypt(nonce, plaintext, aad)


def aead_decrypt(key32: bytes, ciphertext: bytes, *, aad: bytes, nonce: bytes) -> bytes:
    if len(key32) != 32:
        raise ValueError("ChaCha20-Poly1305 key must be 32 bytes")
    aead = ChaCha20Poly1305(key32)
    return aead.decrypt(nonce, ciphertext, aad)
