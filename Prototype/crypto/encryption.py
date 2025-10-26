"""
AES-GCM encryption for model payloads
Provides authenticated encryption with associated data (AEAD)
"""

import os
from typing import Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend


class AESGCMEncryption:
    """
    AES-256-GCM encryption for model payloads
    Provides confidentiality and integrity
    """
    
    def __init__(self, key_size: int = 256):
        """
        Initialize AES-GCM encryption
        
        Args:
            key_size: Key size in bits (128, 192, or 256)
        """
        if key_size not in [128, 192, 256]:
            raise ValueError("Key size must be 128, 192, or 256 bits")
        
        self.key_size = key_size
        self.key_bytes = key_size // 8
    
    def encrypt(
        self,
        plaintext: bytes,
        key: bytes,
        associated_data: bytes = b""
    ) -> Tuple[bytes, bytes]:
        """
        Encrypt data with AES-GCM
        
        Args:
            plaintext: Data to encrypt
            key: Encryption key (must be appropriate size)
            associated_data: Additional authenticated data (optional)
        
        Returns:
            (ciphertext, nonce)
        """
        # Ensure key is correct length
        if len(key) < self.key_bytes:
            raise ValueError(f"Key must be at least {self.key_bytes} bytes")
        
        key = key[:self.key_bytes]
        
        # Generate random nonce
        nonce = os.urandom(12)  # 96 bits recommended for GCM
        
        # Create AESGCM cipher
        aesgcm = AESGCM(key)
        
        # Encrypt
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)
        
        return ciphertext, nonce
    
    def decrypt(
        self,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        associated_data: bytes = b""
    ) -> bytes:
        """
        Decrypt data with AES-GCM
        
        Args:
            ciphertext: Encrypted data
            key: Decryption key
            nonce: Nonce used during encryption
            associated_data: Additional authenticated data (must match encryption)
        
        Returns:
            Decrypted plaintext
        
        Raises:
            cryptography.exceptions.InvalidTag: If authentication fails
        """
        # Ensure key is correct length
        if len(key) < self.key_bytes:
            raise ValueError(f"Key must be at least {self.key_bytes} bytes")
        
        key = key[:self.key_bytes]
        
        # Create AESGCM cipher
        aesgcm = AESGCM(key)
        
        # Decrypt and verify
        plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data)
        
        return plaintext
    
    @staticmethod
    def generate_key(key_size: int = 256) -> bytes:
        """
        Generate a random encryption key
        
        Args:
            key_size: Key size in bits
        
        Returns:
            Random key
        """
        return os.urandom(key_size // 8)
