# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import json
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.models.neural_net import SimpleNN
from src.models.model_utils import create_synthetic_dataset, train_model, evaluate_model
from src.verification.prover import ModelProver
from src.verification.verifier import ModelVerifier

def create_training_progress_plot(losses, accuracies):
    fig = go.Figure()
    
    # Add traces for loss and accuracy
    fig.add_trace(go.Scatter(y=losses, name="Training Loss", line=dict(color="red")))
    fig.add_trace(go.Scatter(y=accuracies, name="Accuracy", line=dict(color="blue")))
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Value",
        hovermode='x unified'
    )
    return fig

def main():
    st.set_page_config(page_title="ML Model Verification System", layout="wide")
    
    st.title("Privacy-Preserving ML Model Verification System")
    
    # Introduction
    st.markdown("""
    ## About This Project
    This system demonstrates how to verify machine learning model performance while preserving privacy using:
    1. **Zero-Knowledge Proofs**: Prove properties about data without revealing the data
    2. **Cryptographic Commitments**: Create tamper-proof commitments to model parameters
    3. **Digital Signatures**: Ensure the authenticity of verification proofs
    
    ### How It Works
    1. Train a neural network model on your data
    2. Generate a proof of the model's performance
    3. Verify the proof without accessing the original model or data
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Model Parameters")
    input_size = st.sidebar.slider("Input Size", 2, 50, 10)
    hidden_size = st.sidebar.slider("Hidden Layer Size", 5, 100, 20)
    output_size = st.sidebar.slider("Output Size (Classes)", 2, 10, 2)
    num_samples = st.sidebar.slider("Number of Training Samples", 100, 5000, 1000)
    batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
    num_epochs = st.sidebar.slider("Number of Epochs", 5, 50, 10)
    
    # Main content
    tabs = st.tabs(["Train Model", "Generate Proof", "Verify Proof"])
    
    with tabs[0]:
        st.header("Train Model")
        if st.button("Train New Model"):
            # Create progress container
            progress_container = st.empty()
            plot_container = st.empty()
            metrics_container = st.empty()
            
            # Training setup
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train, y_train = create_synthetic_dataset(input_size, num_samples, output_size)
            X_test, y_test = create_synthetic_dataset(input_size, num_samples//5, output_size)
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
            
            model = SimpleNN(input_size, hidden_size, output_size)
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters())
            
            # Training loop with progress
            losses = []
            accuracies = []
            
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                # Calculate accuracy
                accuracy = evaluate_model(model, test_loader)
                losses.append(running_loss / len(train_loader))
                accuracies.append(accuracy)
                
                # Update progress
                progress_container.progress((epoch + 1) / num_epochs)
                plot_container.plotly_chart(create_training_progress_plot(losses, accuracies))
                metrics_container.metric("Current Accuracy", f"{accuracy:.4f}")
            
            # Save model and data loaders in session state
            st.session_state['model'] = model
            st.session_state['test_loader'] = test_loader
            st.session_state['final_accuracy'] = accuracy
            
            st.success("Training completed!")
    
    with tabs[1]:
        st.header("Generate Proof")
        if 'model' not in st.session_state:
            st.warning("Please train a model first!")
        else:
            if st.button("Generate Proof"):
                with st.spinner("Generating zero-knowledge proof..."):
                    # Create proof
                    prover = ModelProver()
                    claimed_accuracy = st.session_state['final_accuracy']
                    proof = prover.create_proof(
                        st.session_state['model'],
                        st.session_state['test_loader'],
                        claimed_accuracy
                    )
                    
                    # Save proof
                    st.session_state['proof'] = proof
                    
                    # Display proof details
                    st.json(proof['proof_data'])
                    st.success("Proof generated successfully!")
    
    with tabs[2]:
        st.header("Verify Proof")
        if 'proof' not in st.session_state:
            st.warning("Please generate a proof first!")
        else:
            if st.button("Verify Proof"):
                with st.spinner("Verifying proof..."):
                    # Verify proof
                    verifier = ModelVerifier()
                    is_valid, message = verifier.verify_proof(st.session_state['proof'])
                    
                    if is_valid:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
    
    # Additional information
    st.markdown("""
    ## Technical Details
    
    ### Privacy Guarantees
    - Model architecture and weights remain private
    - Training and test data remain confidential
    - Only performance metrics are revealed
    
    ### Security Features
    - RSA-2048 for digital signatures
    - SHA-256 for cryptographic commitments
    - Zero-knowledge proofs for privacy preservation
    
    ### Use Cases
    1. ML model auditing
    2. Performance verification
    3. Regulatory compliance
    4. Competitive model benchmarking
    """)

if __name__ == "__main__":
    main()