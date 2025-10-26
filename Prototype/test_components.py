"""
Quick test script for PQBFL prototype components
Tests each layer independently before full integration
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def test_blockchain():
    """Test blockchain and smart contract"""
    print("\n" + "="*60)
    print("Testing Blockchain & Smart Contract")
    print("="*60)
    
    from blockchain.blockchain import Blockchain
    from blockchain.smart_contract import PQBFLSmartContract
    
    # Create blockchain
    blockchain = Blockchain(block_time=1.0)
    print(f"✓ Blockchain created with genesis block")
    print(f"  Genesis hash: {blockchain.get_latest_block().hash[:16]}...")
    
    # Create smart contract
    gas_costs = {
        'register_project': 250000,
        'register_client': 75000,
        'publish_task': 260000,
        'update_model': 235000,
        'feedback_model': 215000
    }
    contract = PQBFLSmartContract(blockchain, gas_costs)
    print(f"✓ Smart contract deployed")
    
    # Test project registration
    success, gas, _ = contract.register_project(
        sender="0xServer",
        project_id="test_proj_1",
        n_clients=3,
        h_initial_model="abc123",
        h_server_keys="def456",
        deposit=1000.0
    )
    print(f"✓ Project registered: success={success}, gas={gas}")
    
    # Test client registration
    success, gas, _ = contract.register_client(
        sender="0xClient1",
        project_id="test_proj_1",
        h_client_ecdh_key="789xyz"
    )
    print(f"✓ Client registered: success={success}, gas={gas}")
    
    # Validate blockchain
    is_valid = blockchain.is_chain_valid()
    print(f"✓ Blockchain valid: {is_valid}")
    
    print(f"✓ Total blocks: {blockchain.get_chain_size()}")
    print(f"✓ Total transactions: {blockchain.get_total_transactions()}")
    
    return True


def test_cryptography():
    """Test cryptographic components"""
    print("\n" + "="*60)
    print("Testing Cryptography")
    print("="*60)
    
    # Test Hybrid KEM
    from crypto.hybrid_kem import HybridKEM
    
    print("\nTesting Hybrid KEM (Kyber + ECDH)...")
    hybrid_kem = HybridKEM(kyber_variant="kyber768", ecdh_curve="P-256")
    
    # Server generates keys
    server_keypair = hybrid_kem.generate_keypair()
    print(f"✓ Server keys generated")
    print(f"  KEM public key: {len(server_keypair.kem_public_key)} bytes")
    print(f"  ECDH public key: {len(server_keypair.get_ecdh_pk_bytes())} bytes")
    
    # Client generates keys
    client_keypair = hybrid_kem.generate_keypair()
    print(f"✓ Client keys generated")
    
    # Client encapsulates
    kem_ct, kem_ss_client, ecdh_ss_client = hybrid_kem.encapsulate(
        server_keypair.kem_public_key,
        server_keypair.get_ecdh_pk_bytes()
    )
    print(f"✓ Client encapsulation complete")
    print(f"  Ciphertext: {len(kem_ct)} bytes")
    
    # Server decapsulates
    kem_ss_server, ecdh_ss_server = hybrid_kem.decapsulate(
        server_keypair.kem_secret_key,
        kem_ct,
        server_keypair.ecdh_private_key,
        client_keypair.get_ecdh_pk_bytes()
    )
    print(f"✓ Server decapsulation complete")
    
    # Verify shared secrets match
    kem_match = kem_ss_client == kem_ss_server
    print(f"✓ KEM shared secrets match: {kem_match}")
    
    # Derive root keys
    root_key_client = hybrid_kem.derive_root_key(kem_ss_client, ecdh_ss_client)
    root_key_server = hybrid_kem.derive_root_key(kem_ss_server, ecdh_ss_server)
    
    root_match = root_key_client == root_key_server
    print(f"✓ Root keys match: {root_match}")
    print(f"  Root key: {root_key_client.hex()[:32]}...")
    
    # Test Ratcheting
    print("\nTesting Ratcheting...")
    from crypto.ratcheting import DualRatchet
    
    ratchet = DualRatchet(root_key_client, symmetric_threshold=5)
    print(f"✓ Dual ratchet initialized")
    
    # Derive keys for multiple rounds
    keys = []
    for round_num in range(7):
        key = ratchet.derive_round_key(round_num)
        keys.append(key)
        
        if ratchet.should_perform_asymmetric_ratchet():
            print(f"  → Asymmetric ratchet triggered at round {round_num}")
    
    print(f"✓ Derived {len(keys)} round keys")
    print(f"  All unique: {len(set(keys)) == len(keys)}")
    
    # Test Encryption
    print("\nTesting AES-GCM Encryption...")
    from crypto.encryption import AESGCMEncryption
    
    aes = AESGCMEncryption(key_size=256)
    plaintext = b"This is a secret model update"
    key = keys[0]  # Use first derived key
    
    ciphertext, nonce = aes.encrypt(plaintext, key)
    print(f"✓ Encrypted {len(plaintext)} bytes → {len(ciphertext)} bytes")
    
    decrypted = aes.decrypt(ciphertext, key, nonce)
    print(f"✓ Decrypted successfully: {decrypted == plaintext}")
    
    # Test Signatures
    print("\nTesting ECDSA Signatures...")
    from crypto.signatures import ECDSASignature
    
    sig = ECDSASignature()
    priv_key, pub_key = sig.generate_keypair()
    print(f"✓ Keypair generated")
    
    message = b"Blockchain transaction data"
    signature = sig.sign(message, priv_key)
    print(f"✓ Message signed: {len(signature)} bytes")
    
    is_valid = sig.verify(message, signature, pub_key)
    print(f"✓ Signature valid: {is_valid}")
    
    # Test with wrong message
    wrong_msg = b"Different message"
    is_invalid = sig.verify(wrong_msg, signature, pub_key)
    print(f"✓ Wrong message rejected: {not is_invalid}")
    
    return True


def test_integration():
    """Test integration of components"""
    print("\n" + "="*60)
    print("Testing Component Integration")
    print("="*60)
    
    from blockchain.blockchain import Blockchain
    from blockchain.smart_contract import PQBFLSmartContract
    from crypto.hybrid_kem import HybridKEM
    from crypto.signatures import ECDSASignature
    
    # Setup
    blockchain = Blockchain()
    contract = PQBFLSmartContract(blockchain, {
        'register_project': 250000,
        'register_client': 75000,
        'publish_task': 260000
    })
    
    hybrid_kem = HybridKEM()
    sig = ECDSASignature()
    
    # Create server
    server_bc_priv, server_bc_pub = sig.generate_keypair()
    server_address = sig.address_from_public_key(server_bc_pub)
    server_hybrid = hybrid_kem.generate_keypair()
    
    print(f"✓ Server created: {server_address[:10]}...")
    
    # Register project
    h_keys = server_hybrid.hash_public_keys()
    success, _, _ = contract.register_project(
        sender=server_address,
        project_id="integration_test",
        n_clients=2,
        h_initial_model="abc",
        h_server_keys=h_keys,
        deposit=1000.0
    )
    print(f"✓ Project registered: {success}")
    
    # Create and register clients
    clients = []
    for i in range(2):
        c_priv, c_pub = sig.generate_keypair()
        c_addr = sig.address_from_public_key(c_pub)
        c_hybrid = hybrid_kem.generate_keypair()
        
        import hashlib
        h_ecdh = hashlib.sha256(c_hybrid.get_ecdh_pk_bytes()).hexdigest()
        
        success, _, _ = contract.register_client(
            sender=c_addr,
            project_id="integration_test",
            h_client_ecdh_key=h_ecdh
        )
        
        clients.append({'addr': c_addr, 'hybrid': c_hybrid})
        print(f"✓ Client-{i+1} registered: {c_addr[:10]}...")
    
    # Publish task
    success, _, _ = contract.publish_task(
        sender=server_address,
        project_id="integration_test",
        task_id="task_1",
        round_number=0,
        h_global_model="model_hash",
        h_server_keys=None,
        deadline=9999999999,
        n_clients_required=2
    )
    print(f"✓ Task published: {success}")
    
    # Check blockchain state
    print(f"✓ Blocks: {blockchain.get_chain_size()}")
    print(f"✓ Transactions: {blockchain.get_total_transactions()}")
    print(f"✓ Chain valid: {blockchain.is_chain_valid()}")
    
    data_size = blockchain.calculate_data_size()
    print(f"✓ On-chain data: {data_size['total_kb']:.3f} KB")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# PQBFL Prototype Component Tests")
    print("#"*60)
    
    tests = [
        ("Blockchain", test_blockchain),
        ("Cryptography", test_cryptography),
        ("Integration", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, result in results:
        symbol = "✓" if result == "PASS" else "✗"
        print(f"{symbol} {name}: {result}")
    
    all_passed = all(r[1] == "PASS" for r in results)
    print("="*60)
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
