from ..crypto.zkp import ZKProof
from torch.utils.data import DataLoader

class ModelProver:
    def __init__(self):
        self.zkp = ZKProof()
    
    def create_proof(self, model, test_loader, claimed_accuracy):
        """Create a proof of model performance"""
        # Create proof data
        proof_data = self.zkp.create_proof_data(
            model=model,
            test_loader=test_loader,
            claimed_accuracy=claimed_accuracy
        )
        
        # Sign the proof
        proof = self.zkp.sign_proof(proof_data)
        
        return proof