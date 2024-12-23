
from Crypto.Hash import SHA256
import json
import numpy as np

class ModelCommitment:
    @staticmethod
    def create_commitment(model_params):
        """Create a commitment to model parameters"""
        hasher = SHA256.new()
        
        # Sort parameters by name for consistency
        for name, param in sorted(model_params.items()):
            param_bytes = param.cpu().numpy().tobytes()
            hasher.update(param_bytes)
        
        return hasher.hexdigest()
    
    @staticmethod
    def verify_commitment(commitment, model_params):
        """Verify a model commitment"""
        new_commitment = ModelCommitment.create_commitment(model_params)
        return commitment == new_commitment