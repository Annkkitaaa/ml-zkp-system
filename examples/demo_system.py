# examples/demo_system.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json

from src.models.neural_net import SimpleNN
from src.models.model_utils import create_synthetic_dataset, train_model, evaluate_model
from src.verification.prover import ModelProver
from src.verification.verifier import ModelVerifier

def run_demo():
    # Parameters
    input_size = 10
    hidden_size = 20
    output_size = 2
    num_samples = 1000
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create synthetic dataset
    print("Creating synthetic dataset...")
    X_train, y_train = create_synthetic_dataset(input_size, num_samples, output_size)
    X_test, y_test = create_synthetic_dataset(input_size, num_samples//5, output_size)

    # Create data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size
    )

    # Create and train model
    print("\nTraining model...")
    model = SimpleNN(input_size, hidden_size, output_size)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    
    model = train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Evaluate model
    print("\nEvaluating model...")
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Model accuracy: {accuracy:.4f}")

    # Create proof
    print("\nGenerating zero-knowledge proof...")
    prover = ModelProver()
    claimed_accuracy = accuracy  # In real scenario, this would be the claimed accuracy
    proof = prover.create_proof(model, test_loader, claimed_accuracy)

    # Save proof to file
    with open('proof.json', 'w') as f:
        json.dump(proof, f, indent=2)
    print("Proof saved to proof.json")

    # Verify proof
    print("\nVerifying proof...")
    verifier = ModelVerifier()
    is_valid, message = verifier.verify_proof(proof)
    print(f"Verification result: {is_valid}")
    print(f"Verification message: {message}")

if __name__ == "__main__":
    run_demo()