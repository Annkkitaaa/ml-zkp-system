import pytest
from src.verification.verifier import ModelVerifier
from src.crypto.zkp import ZKProof

def test_verifier():
    # Create a valid proof
    zkp = ZKProof()
    proof_data = {
        "model_commitment": "dummy_commitment",
        "claimed_accuracy": 0.85,
        "actual_accuracy": 0.85,
        "test_set_size": 100
    }
    proof = zkp.sign_proof(proof_data)
    
    # Verify the proof
    verifier = ModelVerifier()
    is_valid, message = verifier.verify_proof(proof)
    assert is_valid
    assert message == "Proof verified successfully"
    
    # Test invalid proof (modified accuracy)
    proof["proof_data"]["actual_accuracy"] = 0.75
    is_valid, message = verifier.verify_proof(proof)
    assert not is_valid