import pytest
from src.crypto.commitment import ModelCommitment
from src.crypto.zkp import ZKProof
import torch

def test_model_commitment():
    # Create dummy model parameters
    params = {
        'layer1.weight': torch.randn(10, 10),
        'layer1.bias': torch.randn(10)
    }
    
    commitment = ModelCommitment.create_commitment(params)
    assert isinstance(commitment, str)
    assert len(commitment) == 64  # SHA-256 hex digest length

def test_zkp():
    zkp = ZKProof()
    
    # Create dummy proof data
    proof_data = {
        "model_commitment": "dummy_commitment",
        "claimed_accuracy": 0.85,
        "actual_accuracy": 0.84,
        "test_set_size": 100
    }
    
    proof = zkp.sign_proof(proof_data)
    assert "proof_data" in proof
    assert "signature" in proof
    assert "public_key" in proof
    
    # Verify signature
    assert ZKProof.verify_signature(proof)