from __future__ import annotations

import hmac
import hashlib
from dataclasses import dataclass


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    if length <= 0:
        raise ValueError("length must be > 0")

    hash_len = hashlib.sha256().digest_size
    n = (length + hash_len - 1) // hash_len
    if n > 255:
        raise ValueError("length too large")

    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]


def kdf_a_root_key(ss_k: bytes, ss_e: bytes, salt: bytes = b"\x00") -> bytes:
    """Asymmetric ratchet KDF_A.

    Matches the paper's description: successively feed (SS_k, SS_e) into a KDF with an all-zero salt.
    Returns a 32-byte root key RK_j.
    """

    prk1 = hkdf_extract(salt=salt, ikm=ss_k)
    prk2 = hkdf_extract(salt=prk1, ikm=ss_e)
    return hkdf_expand(prk=prk2, info=b"pqbfl:RK", length=32)


@dataclass
class SymmetricRatchetState:
    chain_key: bytes
    index: int = 0


def kdf_s_next(state: SymmetricRatchetState) -> tuple[SymmetricRatchetState, bytes]:
    """Symmetric ratchet KDF_S.

    Given a chain key CK_{i,j}, derives (CK_{i+1,j}, K_{i,j}).
    We use HMAC-SHA256 as the PRF.
    """

    ck = state.chain_key
    next_ck = hmac.new(ck, b"pqbfl:CK", hashlib.sha256).digest()
    model_key = hmac.new(ck, b"pqbfl:MK", hashlib.sha256).digest()
    return SymmetricRatchetState(chain_key=next_ck, index=state.index + 1), model_key


def chain_key_from_root(root_key: bytes) -> SymmetricRatchetState:
    ck0 = hmac.new(root_key, b"pqbfl:CK0", hashlib.sha256).digest()
    return SymmetricRatchetState(chain_key=ck0, index=0)
