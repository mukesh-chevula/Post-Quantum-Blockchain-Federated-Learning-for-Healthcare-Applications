"""
ECDSA signatures for blockchain transactions and off-chain messages
"""

import hashlib
from typing import Tuple
from ecdsa import SigningKey, VerifyingKey, SECP256k1, BadSignatureError


class ECDSASignature:
    """
    ECDSA signature scheme using secp256k1 curve
    Used for blockchain transactions and message authentication
    """
    
    def __init__(self, curve=SECP256k1):
        """
        Initialize ECDSA signature scheme
        
        Args:
            curve: Elliptic curve to use (default: SECP256k1)
        """
        self.curve = curve
    
    def generate_keypair(self) -> Tuple[SigningKey, VerifyingKey]:
        """
        Generate a new ECDSA key pair
        
        Returns:
            (private_key, public_key)
        """
        private_key = SigningKey.generate(curve=self.curve)
        public_key = private_key.get_verifying_key()
        
        return (private_key, public_key)  # type: ignore[return-value]
    
    def sign(self, message: bytes, private_key: SigningKey) -> bytes:
        """
        Sign a message
        
        Args:
            message: Message to sign
            private_key: Signing key
        
        Returns:
            Signature bytes
        """
        # Hash the message first (deterministic)
        message_hash = hashlib.sha256(message).digest()
        
        # Sign the hash
        signature = private_key.sign_digest(
            message_hash,
            sigencode=lambda r, s, order: r.to_bytes(32, 'big') + s.to_bytes(32, 'big')
        )
        
        return signature
    
    def verify(self, message: bytes, signature: bytes, public_key: VerifyingKey) -> bool:
        """
        Verify a signature
        
        Args:
            message: Original message
            signature: Signature to verify
            public_key: Public key for verification
        
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Hash the message
            message_hash = hashlib.sha256(message).digest()
            
            # Decode function for signature
            def decode_signature(sig, order):
                r = int.from_bytes(sig[:32], 'big')
                s = int.from_bytes(sig[32:], 'big')
                return r, s
            
            # Verify
            public_key.verify_digest(
                signature,
                message_hash,
                sigdecode=decode_signature
            )
            return True
        except BadSignatureError:
            return False
    
    @staticmethod
    def private_key_to_bytes(private_key: SigningKey) -> bytes:
        """Convert private key to bytes"""
        return private_key.to_string()
    
    @staticmethod
    def private_key_from_bytes(key_bytes: bytes, curve=SECP256k1) -> SigningKey:
        """Load private key from bytes"""
        return SigningKey.from_string(key_bytes, curve=curve)
    
    @staticmethod
    def public_key_to_bytes(public_key: VerifyingKey) -> bytes:
        """Convert public key to bytes"""
        return public_key.to_string()
    
    @staticmethod
    def public_key_from_bytes(key_bytes: bytes, curve=SECP256k1) -> VerifyingKey:
        """Load public key from bytes"""
        return VerifyingKey.from_string(key_bytes, curve=curve)
    
    @staticmethod
    def address_from_public_key(public_key: VerifyingKey) -> str:
        """
        Generate blockchain address from public key
        Similar to Ethereum address derivation
        
        Args:
            public_key: Public key
        
        Returns:
            Address as hex string (0x...)
        """
        pub_bytes = public_key.to_string()
        # Hash the public key
        hash_obj = hashlib.sha256(pub_bytes).digest()
        # Take last 20 bytes and convert to hex
        address = "0x" + hash_obj[-20:].hex()
        return address
