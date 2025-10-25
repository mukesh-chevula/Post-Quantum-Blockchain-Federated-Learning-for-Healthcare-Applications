"""
Simulated blockchain for PQBFL prototype
Implements a simple proof-of-authority blockchain for demonstration
"""

import hashlib
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

from .transactions import Transaction


@dataclass
class Block:
    """Blockchain block"""
    index: int
    timestamp: float
    transactions: List[Transaction]
    previous_hash: str
    nonce: int = 0
    hash: str = field(init=False)
    
    def __post_init__(self):
        self.hash = self.compute_hash()
    
    def compute_hash(self) -> str:
        """Compute block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> dict:
        """Convert block to dictionary"""
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash
        }


class Blockchain:
    """
    Simulated blockchain for PQBFL
    Implements basic blockchain functionality with event emission
    """
    
    def __init__(self, block_time: float = 2.0):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.block_time = block_time
        self.last_block_time = time.time()
        
        # Event listeners
        self.event_listeners: Dict[str, List] = {}
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=[],
            previous_hash="0" * 64
        )
        self.chain.append(genesis_block)
    
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add a transaction to pending transactions
        Returns True if successfully added
        """
        # In a real blockchain, we would validate the signature here
        self.pending_transactions.append(transaction)
        
        # Emit event
        self._emit_event(transaction.tx_type, transaction)
        
        # Auto-mine if enough time has passed
        if time.time() - self.last_block_time >= self.block_time:
            self.mine_pending_transactions()
        
        return True
    
    def mine_pending_transactions(self) -> Optional[Block]:
        """
        Mine a new block with pending transactions
        In a real blockchain, this would involve proof-of-work
        """
        if not self.pending_transactions:
            return None
        
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions.copy(),
            previous_hash=previous_block.hash
        )
        
        self.chain.append(new_block)
        self.pending_transactions = []
        self.last_block_time = time.time()
        
        return new_block
    
    def get_latest_block(self) -> Block:
        """Get the most recent block"""
        return self.chain[-1]
    
    def is_chain_valid(self) -> bool:
        """Validate the entire blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is correct
            if current_block.hash != current_block.compute_hash():
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_transactions_by_type(self, tx_type: str) -> List[Transaction]:
        """Get all transactions of a specific type"""
        transactions = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_type == tx_type:
                    transactions.append(tx)
        return transactions
    
    def get_transactions_by_address(self, address: str) -> List[Transaction]:
        """Get all transactions from a specific address"""
        transactions = []
        for block in self.chain:
            for tx in block.transactions:
                if tx.sender == address:
                    transactions.append(tx)
        return transactions
    
    def get_transaction_by_hash(self, tx_hash: str) -> Optional[Transaction]:
        """Find a transaction by its hash"""
        for block in self.chain:
            for tx in block.transactions:
                if tx.tx_hash == tx_hash:
                    return tx
        return None
    
    def subscribe_event(self, event_type: str, callback):
        """Subscribe to blockchain events"""
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
        self.event_listeners[event_type].append(callback)
    
    def _emit_event(self, event_type: str, data):
        """Emit an event to all subscribers"""
        if event_type in self.event_listeners:
            for callback in self.event_listeners[event_type]:
                callback(data)
    
    def get_chain_size(self) -> int:
        """Get the total number of blocks"""
        return len(self.chain)
    
    def get_total_transactions(self) -> int:
        """Get the total number of transactions across all blocks"""
        return sum(len(block.transactions) for block in self.chain)
    
    def export_chain(self) -> List[dict]:
        """Export the entire blockchain as a list of dictionaries"""
        return [block.to_dict() for block in self.chain]
    
    def calculate_data_size(self) -> Dict[str, Any]:
        """Calculate on-chain data size in bytes"""
        total_size = 0
        tx_sizes = {}
        
        for block in self.chain:
            block_json = json.dumps(block.to_dict())
            total_size += len(block_json.encode('utf-8'))
            
            for tx in block.transactions:
                tx_type = tx.tx_type
                tx_json = json.dumps(tx.to_dict())
                tx_size = len(tx_json.encode('utf-8'))
                
                if tx_type not in tx_sizes:
                    tx_sizes[tx_type] = 0
                tx_sizes[tx_type] += tx_size
        
        return {
            'total_bytes': total_size,
            'total_kb': total_size / 1024,
            'by_transaction_type': tx_sizes
        }
