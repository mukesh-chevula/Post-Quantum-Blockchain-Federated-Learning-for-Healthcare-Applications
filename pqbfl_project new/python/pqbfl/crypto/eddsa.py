from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


@dataclass(frozen=True)
class Ed25519Keypair:
    private_key: ed25519.Ed25519PrivateKey
    public_key_bytes: bytes


def ed25519_keygen() -> Ed25519Keypair:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return Ed25519Keypair(private_key=private_key, public_key_bytes=public_key_bytes)


def ed25519_sign(private_key: ed25519.Ed25519PrivateKey, message: bytes) -> bytes:
    return private_key.sign(message)


def ed25519_verify(public_key_bytes: bytes, message: bytes, signature: bytes) -> bool:
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
        pub.verify(signature, message)
        return True
    except (ValueError, InvalidSignature):
        return False
