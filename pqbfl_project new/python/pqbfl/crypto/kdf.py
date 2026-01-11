from __future__ import annotations

import hmac
import hashlib
from dataclasses import dataclass


def _kdf_hashmod():
    """Return a hashlib-like module for HMAC (BLAKE3-256 preferred)."""

    try:
        from blake3 import blake3  # type: ignore

        class _Blake3Hash:
            digest_size = 32
            # HMAC uses block_size for ipad/opad. BLAKE3 chunk size is 64 bytes; using 64 is standard here.
            block_size = 64

            def __init__(self, data: bytes = b"") -> None:
                self._h = blake3(data)

            def copy(self) -> "_Blake3Hash":
                other = _Blake3Hash()
                other._h = self._h.copy()
                return other

            def digest(self) -> bytes:
                return self._h.digest(length=32)

            def update(self, data: bytes) -> None:
                self._h.update(data)

        return _Blake3Hash
    except Exception:
        return hashlib.sha256


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, _kdf_hashmod()).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    if length <= 0:
        raise ValueError("length must be > 0")

    hash_len = _kdf_hashmod().digest_size
    n = (length + hash_len - 1) // hash_len
    if n > 255:
        raise ValueError("length too large")

    okm = b""
    t = b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), _kdf_hashmod()).digest()
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
    We use HMAC-(BLAKE3-256) as the PRF (falls back to HMAC-SHA256).
    """

    ck = state.chain_key
    next_ck = hmac.new(ck, b"pqbfl:CK", _kdf_hashmod()).digest()
    model_key = hmac.new(ck, b"pqbfl:MK", _kdf_hashmod()).digest()
    return SymmetricRatchetState(chain_key=next_ck, index=state.index + 1), model_key


def chain_key_from_root(root_key: bytes) -> SymmetricRatchetState:
    ck0 = hmac.new(root_key, b"pqbfl:CK0", _kdf_hashmod()).digest()
    return SymmetricRatchetState(chain_key=ck0, index=0)
