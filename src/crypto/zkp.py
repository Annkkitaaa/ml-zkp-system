# src/crypto/zkp.py

from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

class ZKProof:
    def __init__(self):
        """Initialize with a new key pair"""
        self.key = RSA.generate(2048)
        self.public_key = self.key.publickey()
    
    def create_proof_data(self, model, test_loader, claimed_accuracy):
        """Create the proof data structure"""
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        actual_accuracy = correct / total if total > 0 else 0
        
        # Get model parameters
        params = {}
        for name, param in model.named_parameters():
            params[name] = param.detach().cpu()
        
        # Create model commitment
        from .commitment import ModelCommitment
        commitment = ModelCommitment.create_commitment(params)
        
        # Create proof structure
        proof_data = {
            "model_commitment": commitment,
            "claimed_accuracy": float(claimed_accuracy),
            "actual_accuracy": float(actual_accuracy),
            "test_set_size": total,
            "timestamp": np.datetime64('now').astype(str)
        }
        
        return proof_data
    
    def sign_proof(self, proof_data):
        """Sign the proof data"""
        proof_bytes = json.dumps(proof_data, sort_keys=True).encode('utf-8')
        hasher = SHA256.new(proof_bytes)
        signature = pkcs1_15.new(self.key).sign(hasher)
        
        return {
            "proof_data": proof_data,
            "signature": signature.hex(),
            "public_key": self.key.publickey().export_key().decode('utf-8')
        }
    
    @staticmethod
    def verify_signature(proof):
        """Verify the proof signature"""
        try:
            # Reconstruct proof bytes
            proof_bytes = json.dumps(proof["proof_data"], sort_keys=True).encode('utf-8')
            hasher = SHA256.new(proof_bytes)
            
            # Import public key
            public_key = RSA.import_key(proof["public_key"].encode('utf-8'))
            
            # Verify signature
            pkcs1_15.new(public_key).verify(
                hasher,
                bytes.fromhex(proof["signature"])
            )
            return True
        except (ValueError, TypeError, KeyError):
            return False