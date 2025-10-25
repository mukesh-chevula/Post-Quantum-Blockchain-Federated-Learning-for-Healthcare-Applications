"""
Transaction types for PQBFL blockchain
Based on the smart contract events defined in the paper
"""

from dataclasses import dataclass, field
from typing import Optional
import time
import hashlib
import json


@dataclass
class Transaction:
    """Base transaction class"""
    tx_type: str
    sender: str  # Blockchain address
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    signature: Optional[bytes] = None
    tx_hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute transaction hash"""
        tx_data = {
            'type': self.tx_type,
            'sender': self.sender,
            'timestamp': self.timestamp,
            'nonce': self.nonce
        }
        tx_string = json.dumps(tx_data, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        """Convert transaction to dictionary"""
        return {
            'tx_type': self.tx_type,
            'sender': self.sender,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'signature': self.signature.hex() if self.signature else None,
            'tx_hash': self.tx_hash
        }


@dataclass
class RegisterProjectTx(Transaction):
    """Register a new FL project
    Event: RegProject(id_p, nClients, sAddr, h_M^0, h_pks)
    """
    tx_type: str = field(default="RegisterProject", init=False)
    project_id: str = ""
    n_clients: int = 0
    h_initial_model: str = ""  # Hash of initial model
    h_server_keys: str = ""  # Hash of concatenated KEM+ECDH public keys
    deposit: float = 0.0
    
    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'project_id': self.project_id,
            'n_clients': self.n_clients,
            'h_initial_model': self.h_initial_model,
            'h_server_keys': self.h_server_keys,
            'deposit': self.deposit
        })
        return base


@dataclass
class RegisterClientTx(Transaction):
    """Register a client to a project
    Event: RegClient(cAddr, id_p, sc, h_epk)
    """
    tx_type: str = field(default="RegisterClient", init=False)
    project_id: str = ""
    h_client_ecdh_key: str = ""  # Hash of client ECDH public key
    initial_score: int = 0
    
    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'project_id': self.project_id,
            'h_client_ecdh_key': self.h_client_ecdh_key,
            'initial_score': self.initial_score
        })
        return base


@dataclass
class PublishTaskTx(Transaction):
    """Publish a new training task/round
    Event: Task(r, h_M^r, h_pks^r, id_p, id_t, nClients, D_t, time)
    """
    tx_type: str = field(default="PublishTask", init=False)
    project_id: str = ""
    task_id: str = ""
    round_number: int = 0
    h_global_model: str = ""  # Hash of encrypted global model info
    h_server_keys: Optional[str] = None  # Hash of new keys (if asymmetric ratchet)
    deadline: float = 0.0
    n_clients_required: int = 0
    
    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'project_id': self.project_id,
            'task_id': self.task_id,
            'round_number': self.round_number,
            'h_global_model': self.h_global_model,
            'h_server_keys': self.h_server_keys,
            'deadline': self.deadline,
            'n_clients_required': self.n_clients_required
        })
        return base


@dataclass
class UpdateModelTx(Transaction):
    """Submit a local model update
    Event: Update(r, h_m^r, h_c_epk, id_p, id_t, cAddr, time)
    """
    tx_type: str = field(default="UpdateModel", init=False)
    project_id: str = ""
    task_id: str = ""
    round_number: int = 0
    h_local_model: str = ""  # Hash of encrypted local model info
    h_client_keys: Optional[str] = None  # Hash of ct||epk (if asymmetric ratchet)
    
    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'project_id': self.project_id,
            'task_id': self.task_id,
            'round_number': self.round_number,
            'h_local_model': self.h_local_model,
            'h_client_keys': self.h_client_keys
        })
        return base


@dataclass
class FeedbackModelTx(Transaction):
    """Provide feedback on a model update
    Event: Feedback(r, id_p, id_t, h_m^r, h_pks^r, cAddr, sc, T)
    """
    tx_type: str = field(default="FeedbackModel", init=False)
    project_id: str = ""
    task_id: str = ""
    round_number: int = 0
    client_address: str = ""
    score: int = 0  # Reward or penalty
    terminate: bool = False  # T flag
    
    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'project_id': self.project_id,
            'task_id': self.task_id,
            'round_number': self.round_number,
            'client_address': self.client_address,
            'score': self.score,
            'terminate': self.terminate
        })
        return base


@dataclass
class TerminateProjectTx(Transaction):
    """Terminate an FL project
    Event: ProjectTerminate(r, id_p, id_t, time)
    """
    tx_type: str = field(default="TerminateProject", init=False)
    project_id: str = ""
    task_id: str = ""
    final_round: int = 0
    
    def __post_init__(self):
        if not self.tx_hash:
            self.tx_hash = self.compute_hash()
    
    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            'project_id': self.project_id,
            'task_id': self.task_id,
            'final_round': self.final_round
        })
        return base
