from __future__ import annotations

from dataclasses import dataclass

from eth_account import Account


HARDHAT_DEFAULT_MNEMONIC = "test test test test test test test test test test test junk"


@dataclass(frozen=True)
class HardhatAccount:
    index: int
    address: str
    private_key_hex: str


def derive_hardhat_account(index: int, mnemonic: str = HARDHAT_DEFAULT_MNEMONIC) -> HardhatAccount:
    if index < 0:
        raise ValueError("index must be >= 0")

    Account.enable_unaudited_hdwallet_features()
    acct = Account.from_mnemonic(mnemonic, account_path=f"m/44'/60'/0'/0/{index}")
    return HardhatAccount(index=index, address=acct.address, private_key_hex=acct.key.hex())
