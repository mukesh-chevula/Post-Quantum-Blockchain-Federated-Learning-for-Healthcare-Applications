"""
Hybrid Key Encapsulation Mechanism
Combines classical ECDH with post-quantum Kyber KEM
Based on the PQBFL protocol specification
"""

import hashlib
import secrets
from typing import Tuple, Optional
from dataclasses import dataclass

# Post-quantum KEM
try:
    from kyber_py.kyber import Kyber512, Kyber768, Kyber1024
except ImportError:
    print("Warning: kyber-py not installed. Install with: pip install kyber-py")
    Kyber512 = Kyber768 = Kyber1024 = None

# Classical ECDH
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend


@dataclass
class HybridKeyPair:
    """Hybrid key pair containing both KEM and ECDH keys"""
    # Kyber KEM keys
    kem_public_key: bytes
    kem_secret_key: bytes
    
    # ECDH keys
    ecdh_private_key: ec.EllipticCurvePrivateKey
    ecdh_public_key: ec.EllipticCurvePublicKey
    
    def get_kem_pk(self) -> bytes:
        """Get Kyber public key"""
        return self.kem_public_key
    
    def get_kem_sk(self) -> bytes:
        """Get Kyber secret key"""
        return self.kem_secret_key
    
    def get_ecdh_pk_bytes(self) -> bytes:
        """Get ECDH public key as bytes"""
        from cryptography.hazmat.primitives import serialization
        return self.ecdh_public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
    
    def hash_public_keys(self) -> str:
        """Hash of concatenated public keys (for blockchain)"""
        kem_pk = self.kem_public_key
        ecdh_pk = self.get_ecdh_pk_bytes()
        concatenated = kem_pk + ecdh_pk
        return hashlib.sha256(concatenated).hexdigest()


class HybridKEM:
    """
    Hybrid KEM combining Kyber (post-quantum) and ECDH (classical)
    Provides quantum resistance while maintaining classical security
    """
    
    def __init__(self, kyber_variant: str = "kyber768", ecdh_curve: str = "P-256"):
        """
        Initialize hybrid KEM
        
        Args:
            kyber_variant: "kyber512", "kyber768", or "kyber1024"
            ecdh_curve: "P-256", "P-384", or "secp256k1"
        """
        self.kyber_variant = kyber_variant
        self.ecdh_curve = ecdh_curve
        
        # Select Kyber variant
        if kyber_variant == "kyber512":
            if Kyber512 is None:
                raise ImportError("kyber-py not installed. Install with: pip install kyber-py")
            self.kyber = Kyber512
        elif kyber_variant == "kyber768":
            if Kyber768 is None:
                raise ImportError("kyber-py not installed. Install with: pip install kyber-py")
            self.kyber = Kyber768
        elif kyber_variant == "kyber1024":
            if Kyber1024 is None:
                raise ImportError("kyber-py not installed. Install with: pip install kyber-py")
            self.kyber = Kyber1024
        else:
            raise ValueError(f"Unknown Kyber variant: {kyber_variant}")
        
        # Select ECDH curve
        if ecdh_curve == "P-256":
            self.curve = ec.SECP256R1()
        elif ecdh_curve == "P-384":
            self.curve = ec.SECP384R1()
        elif ecdh_curve == "secp256k1":
            self.curve = ec.SECP256K1()
        else:
            raise ValueError(f"Unknown ECDH curve: {ecdh_curve}")
    
    def generate_keypair(self) -> HybridKeyPair:
        """
        Generate a hybrid key pair (KEM + ECDH)
        
        Returns:
            HybridKeyPair containing both KEM and ECDH keys
        """
        # Generate Kyber KEM key pair
        kem_pk, kem_sk = self.kyber.keygen()
        
        # Generate ECDH key pair
        ecdh_private_key = ec.generate_private_key(self.curve, default_backend())
        ecdh_public_key = ecdh_private_key.public_key()
        
        return HybridKeyPair(
            kem_public_key=kem_pk,
            kem_secret_key=kem_sk,
            ecdh_private_key=ecdh_private_key,
            ecdh_public_key=ecdh_public_key
        )
    
    def encapsulate(
        self,
        kem_public_key: bytes,
        ecdh_public_key_bytes: bytes
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encapsulate: generate shared secrets from public keys
        
        Args:
            kem_public_key: Recipient's Kyber public key
            ecdh_public_key_bytes: Recipient's ECDH public key (serialized)
        
        Returns:
            (kem_ciphertext, kem_shared_secret, ecdh_shared_secret)
        """
        # Kyber encapsulation
        kem_shared_secret, kem_ciphertext = self.kyber.encaps(kem_public_key)
        
        # ECDH key exchange (generate ephemeral key)
        ecdh_ephemeral_private = ec.generate_private_key(self.curve, default_backend())
        
        # Load recipient's ECDH public key
        from cryptography.hazmat.primitives import serialization
        ecdh_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            self.curve,
            ecdh_public_key_bytes
        )
        
        # Perform ECDH
        ecdh_shared_secret = ecdh_ephemeral_private.exchange(
            ec.ECDH(),
            ecdh_public_key
        )
        
        return kem_ciphertext, kem_shared_secret, ecdh_shared_secret
    
    def decapsulate(
        self,
        kem_secret_key: bytes,
        kem_ciphertext: bytes,
        ecdh_private_key: ec.EllipticCurvePrivateKey,
        ecdh_public_key_bytes: bytes
    ) -> Tuple[bytes, bytes]:
        """
        Decapsulate: recover shared secrets from ciphertext and keys
        
        Args:
            kem_secret_key: Own Kyber secret key
            kem_ciphertext: Received Kyber ciphertext
            ecdh_private_key: Own ECDH private key
            ecdh_public_key_bytes: Sender's ECDH public key (serialized)
        
        Returns:
            (kem_shared_secret, ecdh_shared_secret)
        """
        # Kyber decapsulation
        kem_shared_secret = self.kyber.decaps(kem_secret_key, kem_ciphertext)
        
        # ECDH key exchange
        from cryptography.hazmat.primitives import serialization
        ecdh_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            self.curve,
            ecdh_public_key_bytes
        )
        
        ecdh_shared_secret = ecdh_private_key.exchange(
            ec.ECDH(),
            ecdh_public_key
        )
        
        return kem_shared_secret, ecdh_shared_secret
    
    def derive_root_key(
        self,
        kem_shared_secret: bytes,
        ecdh_shared_secret: bytes,
        salt: Optional[bytes] = None,
        info: bytes = b"PQBFL_root_key"
    ) -> bytes:
        """
        Derive root key from hybrid shared secrets using HKDF
        
        Args:
            kem_shared_secret: Shared secret from Kyber KEM
            ecdh_shared_secret: Shared secret from ECDH
            salt: Optional salt (default: zero bytes)
            info: Context information
        
        Returns:
            Derived root key (32 bytes)
        """
        if salt is None:
            salt = b'\x00' * 32
        
        # Concatenate shared secrets
        combined_secret = kem_shared_secret + ecdh_shared_secret
        
        # HKDF-Extract then HKDF-Expand
        hkdf = HKDF(
            algorithm=hashes.SHA384(),
            length=32,  # 256-bit root key
            salt=salt,
            info=info,
            backend=default_backend()
        )
        
        root_key = hkdf.derive(combined_secret)
        
        return root_key
    
    @staticmethod
    def hash_keys_for_blockchain(kem_pk: bytes, ecdh_pk_bytes: bytes) -> str:
        """
        Hash public keys for blockchain storage
        
        Args:
            kem_pk: Kyber public key
            ecdh_pk_bytes: ECDH public key bytes
        
        Returns:
            SHA-256 hash as hex string
        """
        concatenated = kem_pk + ecdh_pk_bytes
        return hashlib.sha256(concatenated).hexdigest()
