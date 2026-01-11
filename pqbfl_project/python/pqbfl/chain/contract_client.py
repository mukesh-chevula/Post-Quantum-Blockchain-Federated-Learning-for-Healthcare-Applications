from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from web3 import Web3


@dataclass(frozen=True)
class PQBFLArtifact:
    abi: list[dict[str, Any]]
    bytecode: str


def load_hardhat_artifact(chain_dir: Path) -> PQBFLArtifact:
    artifact_path = chain_dir / "artifacts" / "contracts" / "PQBFL.sol" / "PQBFL.json"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Hardhat artifact not found at {artifact_path}. Run `npm install` and `npm run compile` in the chain folder."
        )

    data = json.loads(artifact_path.read_text())
    abi = data["abi"]
    bytecode = data["bytecode"]
    return PQBFLArtifact(abi=abi, bytecode=bytecode)


class PQBFLContractClient:
    def __init__(self, w3: Web3, address: str, abi: list[dict[str, Any]]):
        self.w3 = w3
        self.address = Web3.to_checksum_address(address)
        self.contract = w3.eth.contract(address=self.address, abi=abi)

    @staticmethod
    def deploy_from_artifact(w3: Web3, artifact: PQBFLArtifact, deployer: str) -> "PQBFLContractClient":
        deployer = Web3.to_checksum_address(deployer)
        contract = w3.eth.contract(abi=artifact.abi, bytecode=artifact.bytecode)
        tx_hash = contract.constructor().transact({"from": deployer})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return PQBFLContractClient(w3=w3, address=receipt.contractAddress, abi=artifact.abi)

    def register_project(self, *, from_addr: str, id_p: int, n_clients: int, h_m0: bytes, h_pks: bytes, deposit_wei: int):
        return self.contract.functions.registerProject(id_p, n_clients, h_m0, h_pks).transact(
            {"from": Web3.to_checksum_address(from_addr), "value": deposit_wei}
        )

    def register_client(self, *, from_addr: str, h_epk: bytes, id_p: int):
        return self.contract.functions.registerClient(h_epk, id_p).transact({"from": Web3.to_checksum_address(from_addr)})

    def publish_task(self, *, from_addr: str, r: int, h_inf_b: bytes, h_pks_r: bytes, id_t: int, id_p: int, deadline: int):
        return self.contract.functions.publishTask(r, h_inf_b, h_pks_r, id_t, id_p, deadline).transact(
            {"from": Web3.to_checksum_address(from_addr)}
        )

    def update_model(self, *, from_addr: str, r: int, h_inf_a: bytes, h_ct_epk: bytes, id_t: int, id_p: int):
        return self.contract.functions.updateModel(r, h_inf_a, h_ct_epk, id_t, id_p).transact(
            {"from": Web3.to_checksum_address(from_addr)}
        )

    def feedback_model(
        self,
        *,
        from_addr: str,
        r: int,
        id_t: int,
        id_p: int,
        client_addr: str,
        h_inf_a: bytes,
        h_pks_r: bytes,
        score_delta: int,
        terminate: bool,
    ):
        return self.contract.functions.feedbackModel(
            r,
            id_t,
            id_p,
            Web3.to_checksum_address(client_addr),
            h_inf_a,
            h_pks_r,
            score_delta,
            terminate,
        ).transact({"from": Web3.to_checksum_address(from_addr)})
