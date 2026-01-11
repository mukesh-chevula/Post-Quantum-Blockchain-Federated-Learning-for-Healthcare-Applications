from __future__ import annotations

from dataclasses import dataclass

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


@dataclass(frozen=True)
class ECDHKeypair:
    private_key: ec.EllipticCurvePrivateKey
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
