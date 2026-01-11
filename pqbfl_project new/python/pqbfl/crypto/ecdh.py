from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, x25519


@dataclass(frozen=True)
class ECDHKeypair:
    private_key: ec.EllipticCurvePrivateKey
    public_key_bytes: bytes


@dataclass(frozen=True)
class X25519Keypair:
    private_key: x25519.X25519PrivateKey
    public_key_bytes: bytes


def ecdh_keygen_secp256k1() -> ECDHKeypair:
    private_key = ec.generate_private_key(ec.SECP256K1())
    public_key = private_key.public_key()
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.CompressedPoint,
    )
    return ECDHKeypair(private_key=private_key, public_key_bytes=public_key_bytes)


def ecdh_shared_secret_secp256k1(
    private_key: ec.EllipticCurvePrivateKey,
    peer_public_key_bytes: bytes,
) -> bytes:
    peer_public_key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), peer_public_key_bytes)
    return private_key.exchange(ec.ECDH(), peer_public_key)


def ecdh_keygen_x25519() -> X25519Keypair:
    private_key = x25519.X25519PrivateKey.generate()
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return X25519Keypair(private_key=private_key, public_key_bytes=public_key_bytes)


def ecdh_shared_secret_x25519(private_key: x25519.X25519PrivateKey, peer_public_key_bytes: bytes) -> bytes:
    peer_public_key = x25519.X25519PublicKey.from_public_bytes(peer_public_key_bytes)
    return private_key.exchange(peer_public_key)
