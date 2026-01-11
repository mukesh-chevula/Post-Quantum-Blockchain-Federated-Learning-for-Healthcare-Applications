from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pqbfl.crypto.aead import aead_decrypt, aead_encrypt, nonce_for_round
from pqbfl.crypto.ecdh import X25519Keypair, ecdh_keygen_x25519, ecdh_shared_secret_x25519
from pqbfl.crypto.eddsa import Ed25519Keypair, ed25519_keygen, ed25519_sign, ed25519_verify
from pqbfl.crypto.kdf import SymmetricRatchetState, chain_key_from_root, kdf_a_root_key, kdf_s_next
from pqbfl.crypto.kyber import KyberEncapResult, KyberKeypair, kyber_decap, kyber_encap, kyber_keygen
from pqbfl.utils import json_dumps_canonical, hash32


@dataclass
class ServerKeys:
    sig: Ed25519Keypair
    kem: KyberKeypair
    ecdh: X25519Keypair


@dataclass
class ClientKeys:
    sig: Ed25519Keypair
    ecdh: X25519Keypair


@dataclass
class SessionState:
    root_key: bytes
    ratchet: SymmetricRatchetState
    j: int
    i: int
    L_j: int


def server_generate_keys() -> ServerKeys:
    return ServerKeys(
        sig=ed25519_keygen(),
        kem=kyber_keygen(),
        ecdh=ecdh_keygen_x25519(),
    )


def client_generate_keys() -> ClientKeys:
    return ClientKeys(sig=ed25519_keygen(), ecdh=ecdh_keygen_x25519())


def h_server_pubkeys(kpk_b: bytes, epk_b: bytes) -> bytes:
    return hash32(kpk_b + epk_b)


@dataclass
class OffchainSignedMessage:
    payload: dict[str, Any]
    signature: bytes

    def serialize_for_signing(self) -> bytes:
        return json_dumps_canonical(self.payload).encode("utf-8")


def server_send_pubkeys(server: ServerKeys, *, tx_r: dict[str, Any], id_p: int) -> OffchainSignedMessage:
    payload = {
        "type": "server_pubkeys",
        "id_p": id_p,
        "kpk_b": server.kem.public_key,
        "epk_b": server.ecdh.public_key_bytes,
        "tx_r": tx_r,
    }
    msg_bytes = json_dumps_canonical(payload).encode("utf-8")
    sig = ed25519_sign(server.sig.private_key, msg_bytes)
    return OffchainSignedMessage(payload=payload, signature=sig)


def client_process_server_pubkeys(
    client: ClientKeys,
    *,
    server_sig_pk: bytes,
    signed: OffchainSignedMessage,
    expected_h_pks: bytes,
) -> KyberEncapResult:
    msg_bytes = signed.serialize_for_signing()
    if not ed25519_verify(server_sig_pk, msg_bytes, signed.signature):
        raise ValueError("server signature invalid")

    kpk_b = signed.payload["kpk_b"]
    epk_b = signed.payload["epk_b"]
    if hash32(kpk_b + epk_b) != expected_h_pks:
        raise ValueError("server pubkey hash mismatch")

    # Client derives shared secrets and produces Kyber ciphertext for server.
    ss_e = ecdh_shared_secret_x25519(client.ecdh.private_key, epk_b)
    encap = kyber_encap(kpk_b)
    _ = kdf_a_root_key(encap.shared_secret, ss_e)  # computed again after server confirmation
    return encap


def client_send_epk_and_ct(client: ClientKeys, *, tx_r: dict[str, Any], id_p: int, ct: bytes) -> OffchainSignedMessage:
    payload = {
        "type": "client_epk_ct",
        "id_p": id_p,
        "epk_a": client.ecdh.public_key_bytes,
        "ct": ct,
        "tx_r": tx_r,
    }
    msg_bytes = json_dumps_canonical(payload).encode("utf-8")
    sig = ed25519_sign(client.sig.private_key, msg_bytes)
    return OffchainSignedMessage(payload=payload, signature=sig)


def server_finish_session(
    server: ServerKeys,
    *,
    client_sig_pk: bytes,
    signed: OffchainSignedMessage,
    expected_h_epk_a: bytes,
) -> bytes:
    msg_bytes = signed.serialize_for_signing()
    if not ed25519_verify(client_sig_pk, msg_bytes, signed.signature):
        raise ValueError("client signature invalid")

    epk_a = signed.payload["epk_a"]
    ct = signed.payload["ct"]
    if hash32(epk_a) != expected_h_epk_a:
        raise ValueError("client epk hash mismatch")

    ss_e = ecdh_shared_secret_x25519(server.ecdh.private_key, epk_a)
    ss_k = kyber_decap(ct, server.kem.secret_key)
    return kdf_a_root_key(ss_k, ss_e)


def client_finish_session(client: ClientKeys, *, server_pub: OffchainSignedMessage, encap: KyberEncapResult) -> bytes:
    epk_b = server_pub.payload["epk_b"]
    ss_e = ecdh_shared_secret_x25519(client.ecdh.private_key, epk_b)
    return kdf_a_root_key(encap.shared_secret, ss_e)


def session_from_root(root_key: bytes, *, L_j: int) -> SessionState:
    return SessionState(root_key=root_key, ratchet=chain_key_from_root(root_key), j=1, i=0, L_j=L_j)


def next_model_key(state: SessionState) -> tuple[SessionState, bytes]:
    state.ratchet, model_key = kdf_s_next(state.ratchet)
    state.i += 1
    return state, model_key


def encrypt_round_message(model_key: bytes, *, round_num: int, direction: str, payload: dict[str, Any]) -> bytes:
    plaintext = json_dumps_canonical(payload).encode("utf-8")
    aad = f"pqbfl:{direction}:{round_num}".encode("utf-8")
    nonce = nonce_for_round(round_num, f"{direction}")
    return aead_encrypt(model_key, plaintext, aad=aad, nonce=nonce)


def decrypt_round_message(model_key: bytes, *, round_num: int, direction: str, ciphertext: bytes) -> dict[str, Any]:
    aad = f"pqbfl:{direction}:{round_num}".encode("utf-8")
    nonce = nonce_for_round(round_num, f"{direction}")
    pt = aead_decrypt(model_key, ciphertext, aad=aad, nonce=nonce)
    import json
    from pqbfl.utils import json_loads_bytes

    return json_loads_bytes(json.loads(pt.decode("utf-8")))
