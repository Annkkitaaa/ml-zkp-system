# Privacy-Preserving ML Model Verification System

A system that combines zero-knowledge proofs, cryptographic commitments, and digital signatures to verify machine learning model performance while maintaining privacy and security.

## Project Overview

This project enables organizations to:
- Prove their ML model's performance without revealing the model
- Verify other organizations' model claims without accessing their data
- Maintain privacy of sensitive data and model architecture
- Create tamper-proof performance proofs

## Project Structure
```
ml-zkp-system/
│
├── src/
│   ├── models/
│   │   ├── neural_net.py      # Neural network implementation
│   │   └── model_utils.py     # Utilities for model operations
│   ├── crypto/
│   │   ├── commitment.py      # Cryptographic commitment schemes
│   │   └── zkp.py            # Zero-knowledge proof implementation
│   └── verification/
│       ├── prover.py         # Proof generation
│       └── verifier.py       # Proof verification
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_crypto.py
│   └── test_verification.py
│
├── examples/
│   ├─demo_system.py 
├── saved_proofs/            # Directory for storing proofs
│
├── app.py                   # Streamlit web interface
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Annkkitaaa/ml-zkp-system.git
cd ml-zkp-system
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Real-World Example: Healthcare Use Case

### Scenario:
- Hospital A has developed a disease prediction model
- Hospital B wants to verify the model's performance
- Patient data must remain private

### Hospital A (Model Owner) Code:
```python
from src.models.neural_net import SimpleNN
from src.verification.prover import ModelProver
import torch

# 1. Load your trained model
model = SimpleNN(input_size=10, hidden_size=20, output_size=2)
model.load_state_dict(torch.load('disease_prediction_model.pth'))

# 2. Prepare test data (private patient data)
test_loader = create_test_dataloader(private_patient_data)

# 3. Generate proof
prover = ModelProver()
proof = prover.create_proof(
    model=model,
    test_loader=test_loader,
    claimed_accuracy=0.95
)

# 4. Save proof for sharing
proof_path, signature_path = save_proof_and_signature(proof)
print(f"Proof saved to: {proof_path}")
print(f"Signature saved to: {signature_path}")
```

### Hospital B (Verifier) Code:
```python
from src.verification.verifier import ModelVerifier

# 1. Load received proof
proof = load_proof("received_proof.pkl")

# 2. Verify the proof
verifier = ModelVerifier()
is_valid, message = verifier.verify_proof(proof)

if is_valid:
    print("✅ Model performance verified!")
    print(f"Verified accuracy: {proof['proof_data']['actual_accuracy']}")
    print("The model meets the claimed performance standards.")
else:
    print("❌ Verification failed:", message)
```

## Using the Web Interface
Live demo: https://mlzkpsystem.streamlit.app/

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. For Model Owners:
   - Go to "Train Model" tab
   - Adjust model parameters
   - Train the model
   - Generate proof in "Generate Proof" tab
   - Share the files from saved_proofs/ directory

3. For Verifiers:
   - Go to "Verify Proof" tab
   - Upload received proof file
   - Click "Verify Proof"
   - View verification results

## Security Features

1. Zero-Knowledge Proofs:
   - Prove model performance without revealing model
   - Maintain data privacy
   - Mathematically verifiable claims

2. Cryptographic Commitments:
   - Tamper-proof model parameter commitments
   - Prevent post-generation modifications
   - Secure hash-based verification

3. Digital Signatures:
   - Ensure proof authenticity
   - Verify proof origin
   - Prevent unauthorized modifications

## Best Practices

1. Proof Generation:
   - Use representative test data
   - Keep private keys secure
   - Document proof generation process

2. Proof Verification:
   - Always verify signatures
   - Check timestamp freshness
   - Validate accuracy claims

3. Security:
   - Regular backup of proofs
   - Secure key management
   - Access control implementation

## Project Dependencies
- PyTorch >= 2.0.0
- Streamlit >= 1.22.0
- pycryptodome >= 3.15.0
- numpy >= 1.21.0
- plotly >= 5.13.0

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

 
