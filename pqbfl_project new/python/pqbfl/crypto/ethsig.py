from __future__ import annotations

from dataclasses import dataclass

from eth_account import Account
from eth_account.messages import encode_defunct


@dataclass(frozen=True)
class EthIdentity:
    address: str
    private_key_hex: str


def sign_bytes(private_key_hex: str, message: bytes) -> bytes:
    signed = Account.sign_message(encode_defunct(message), private_key=private_key_hex)
    return signed.signature


def recover_signer(message: bytes, signature: bytes) -> str:
    return Account.recover_message(encode_defunct(message), signature=signature)
