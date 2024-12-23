from ..crypto.zkp import ZKProof
import json

class ModelVerifier:
    @staticmethod
    def verify_proof(proof, accuracy_tolerance=0.01):
        """
        Verify a model proof
        
        Args:
            proof: The proof dictionary containing proof_data and signature
            accuracy_tolerance: Acceptable difference between claimed and actual accuracy
            
        Returns:
            tuple: (is_valid, message)
        """
        # Step 1: Verify signature
        if not ZKProof.verify_signature(proof):
            return False, "Invalid signature"
        
        # Step 2: Verify accuracy claim
        proof_data = proof["proof_data"]
        accuracy_diff = abs(proof_data["claimed_accuracy"] - 
                          proof_data["actual_accuracy"])
        
        if accuracy_diff > accuracy_tolerance:
            return False, (f"Accuracy claim differs from actual accuracy "
                         f"by {accuracy_diff:.4f}")
        
        return True, "Proof verified successfully"