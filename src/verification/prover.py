import torch
from ..crypto.zkp import ZKProof
from ..crypto.commitment import ModelCommitment

class ModelProver:
    def __init__(self):
        self.zkp = ZKProof()
    
    def create_proof(self, model, test_loader, claimed_accuracy):
        """Create a proof of model performance"""
        # Create proof data
        proof_data = self.zkp.create_proof_data(
            model, test_loader, claimed_accuracy
        )
        
        # Sign the proof
        proof = self.zkp.sign_proof(proof_data)
        
        return proof