"""
Ratcheting mechanisms for forward secrecy and post-compromise security
Implements symmetric and asymmetric ratcheting as described in PQBFL paper
"""

from typing import Tuple, Optional
import hashlib

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend


class SymmetricRatchet:
    """
    Symmetric ratcheting for per-round key derivation
    Provides forward secrecy within an asymmetric ratchet period
    """
    
    def __init__(self, root_key: bytes, hash_algorithm=hashes.SHA384()):
        """
        Initialize symmetric ratchet with a root key
        
        Args:
            root_key: Initial root key (from asymmetric ratchet)
            hash_algorithm: Hash algorithm for HKDF
        """
        self.root_key = root_key
        self.hash_algorithm = hash_algorithm
        self.chain_key = None
        self.step = 0
    
    def initialize_chain(self, info: bytes = b"PQBFL_chain_init") -> bytes:
        """
        Initialize the chain key from root key
        
        Args:
            info: Context information
        
        Returns:
            First chain key
        """
        hkdf = HKDF(
            algorithm=self.hash_algorithm,
            length=32,
            salt=None,
            info=info,
            backend=default_backend()
        )
        
        self.chain_key = hkdf.derive(self.root_key)
        self.step = 0
        return self.chain_key
    
    def derive_model_key(self, label: bytes = b"model_key") -> Tuple[bytes, bytes]:
        """
        Derive next model key from current chain key
        
        Args:
            label: Label for this key derivation
        
        Returns:
            (model_key, next_chain_key)
        """
        if self.chain_key is None:
            self.initialize_chain()
        
        # Ensure chain_key is not None
        if self.chain_key is None:
            raise ValueError("Chain key not initialized. Call initialize_chain() first.")
        
        # Derive model key for this round
        hkdf_model = HKDF(
            algorithm=self.hash_algorithm,
            length=32,
            salt=self.chain_key,
            info=label + b"_" + str(self.step).encode(),
            backend=default_backend()
        )
        model_key = hkdf_model.derive(self.chain_key)
        
        # Derive next chain key
        hkdf_chain = HKDF(
            algorithm=self.hash_algorithm,
            length=32,
            salt=self.chain_key,
            info=b"chain_key_" + str(self.step + 1).encode(),
            backend=default_backend()
        )
        next_chain_key = hkdf_chain.derive(self.chain_key)
        
        # Update state
        self.chain_key = next_chain_key
        self.step += 1
        
        return model_key, self.chain_key
    
    def get_step(self) -> int:
        """Get current ratchet step"""
        return self.step


class AsymmetricRatchet:
    """
    Asymmetric ratcheting for root key rotation
    Provides post-compromise security by refreshing root keys
    """
    
    def __init__(self, threshold: int = 10):
        """
        Initialize asymmetric ratchet
        
        Args:
            threshold: Number of symmetric ratchets before triggering asymmetric ratchet
        """
        self.threshold = threshold
        self.ratchet_count = 0
        self.root_keys = []
        self.current_root_key = None
    
    def set_root_key(self, root_key: bytes):
        """
        Set a new root key (from hybrid KEM)
        
        Args:
            root_key: New root key
        """
        self.current_root_key = root_key
        self.root_keys.append(root_key)
        self.ratchet_count += 1
    
    def should_ratchet(self, symmetric_step: int) -> bool:
        """
        Check if asymmetric ratchet should be triggered
        
        Args:
            symmetric_step: Current symmetric ratchet step
        
        Returns:
            True if asymmetric ratchet should occur
        """
        return symmetric_step >= self.threshold
    
    def get_root_key(self) -> Optional[bytes]:
        """Get current root key"""
        return self.current_root_key
    
    def get_ratchet_count(self) -> int:
        """Get number of asymmetric ratchets performed"""
        return self.ratchet_count


class DualRatchet:
    """
    Combined symmetric and asymmetric ratcheting system
    Manages the complete ratcheting mechanism for PQBFL
    """
    
    def __init__(self, initial_root_key: bytes, symmetric_threshold: int = 10):
        """
        Initialize dual ratchet system
        
        Args:
            initial_root_key: Initial root key from session establishment
            symmetric_threshold: L_j parameter (symmetric ratchets per asymmetric ratchet)
        """
        self.asymmetric_ratchet = AsymmetricRatchet(threshold=symmetric_threshold)
        self.asymmetric_ratchet.set_root_key(initial_root_key)
        
        self.symmetric_ratchet = SymmetricRatchet(initial_root_key)
        self.symmetric_ratchet.initialize_chain()
        
        self.round_number = 0
        self.total_symmetric_steps = 0
    
    def derive_round_key(self, round_number: int) -> bytes:
        """
        Derive model key for a specific round
        
        Args:
            round_number: FL training round number
        
        Returns:
            Model key for this round
        """
        model_key, _ = self.symmetric_ratchet.derive_model_key(
            label=f"round_{round_number}".encode()
        )
        
        self.round_number = round_number
        self.total_symmetric_steps = self.symmetric_ratchet.get_step()
        
        return model_key
    
    def should_perform_asymmetric_ratchet(self) -> bool:
        """Check if asymmetric ratchet should be performed"""
        return self.asymmetric_ratchet.should_ratchet(self.total_symmetric_steps)
    
    def perform_asymmetric_ratchet(self, new_root_key: bytes):
        """
        Perform asymmetric ratchet with new root key
        
        Args:
            new_root_key: New root key from fresh hybrid KEM
        """
        # Set new root key
        self.asymmetric_ratchet.set_root_key(new_root_key)
        
        # Reinitialize symmetric ratchet with new root key
        self.symmetric_ratchet = SymmetricRatchet(new_root_key)
        self.symmetric_ratchet.initialize_chain()
        
        # Reset symmetric step counter
        self.total_symmetric_steps = 0
    
    def get_state(self) -> dict:
        """Get current ratchet state for debugging/monitoring"""
        return {
            'round_number': self.round_number,
            'total_symmetric_steps': self.total_symmetric_steps,
            'asymmetric_ratchet_count': self.asymmetric_ratchet.get_ratchet_count(),
            'threshold': self.asymmetric_ratchet.threshold
        }
