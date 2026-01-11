from __future__ import annotations

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pqbfl.utils import sha256


def nonce_for_round(round_num: int, label: str) -> bytes:
    if round_num < 0:
        raise ValueError("round_num must be >= 0")
    seed = f"pqbfl:{label}:{round_num}".encode("utf-8")
    return sha256(seed)[:12]


def aead_encrypt(key32: bytes, plaintext: bytes, *, aad: bytes, nonce: bytes) -> bytes:
    if len(key32) != 32:
        raise ValueError("AES-256-GCM key must be 32 bytes")
    aesgcm = AESGCM(key32)
    return aesgcm.encrypt(nonce, plaintext, aad)


def aead_decrypt(key32: bytes, ciphertext: bytes, *, aad: bytes, nonce: bytes) -> bytes:
    if len(key32) != 32:
        raise ValueError("AES-256-GCM key must be 32 bytes")
    aesgcm = AESGCM(key32)
    return aesgcm.decrypt(nonce, ciphertext, aad)
