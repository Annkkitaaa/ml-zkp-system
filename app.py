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
import os
from datetime import datetime
import pickle
from src.models.neural_net import SimpleNN
from src.models.model_utils import create_synthetic_dataset, train_model, evaluate_model
from src.verification.prover import ModelProver
from src.verification.verifier import ModelVerifier

# Set page configuration
st.set_page_config(
    page_title="ML Model Verification System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f7ff;
        }
        .custom-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            padding: 0 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 10px 20px;
            background-color: white;
            border-radius: 5px;
            color: #0a1e63;
            font-weight: 600;
            border: 1px solid #e6e6e6;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #0a1e63;
            color: white;
            border: none;
        }
        div[data-testid="stSidebarContent"] {
            background-color: #f8f9fa;
        }
        .metric-container {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        .status-box {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            color: #856404;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
    </style>
""", unsafe_allow_html=True)

def create_training_progress_plot(losses, accuracies):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(y=losses, name="Training Loss", 
                            line=dict(color="#e63946", width=2)))
    fig.add_trace(go.Scatter(y=accuracies, name="Accuracy", 
                            line=dict(color="#457b9d", width=2)))
    
    fig.update_layout(
        title={
            'text': "Training Progress",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Epoch",
        yaxis_title="Value",
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def save_proof_and_signature(proof, directory="saved_proofs"):
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"proof_{timestamp}"
    
    # Save proof data
    proof_path = os.path.join(directory, f"{filename}.pkl")
    with open(proof_path, 'wb') as f:
        pickle.dump(proof, f)
    
    # Save signature separately
    signature_path = os.path.join(directory, f"{filename}_signature.txt")
    with open(signature_path, 'w') as f:
        f.write(proof['signature'])
        
    return proof_path, signature_path

def load_proof(proof_path):
    try:
        with open(proof_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading proof: {str(e)}")
        return None

def main():
    # Header
    st.title("🔒 Privacy-Preserving ML Model Verification System")
    
    # About This Project section
    st.header("About This Project")
    st.write("This system demonstrates how to verify machine learning model performance while preserving privacy using cutting-edge cryptographic techniques.")
    
    st.subheader("Key Components:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Zero-Knowledge Proofs (ZKP)**")
        st.write("Prove properties about data without revealing the actual data")
    with col2:
        st.markdown("**2. Cryptographic Commitments**")
        st.write("Create tamper-proof commitments to model parameters")
    with col3:
        st.markdown("**3. Digital Signatures**")
        st.write("Ensure the authenticity of verification proofs")
    
    st.markdown("---")

    # Use Cases Section
    st.header("Real-World Use Cases")
    
    with st.expander("🏥 Healthcare - Disease Prediction Models", expanded=True):
        st.markdown("""
        **Challenge**: Hospitals need to validate AI models while protecting patient data
        
        **Solution**:
        1. Hospital develops disease prediction model using private patient data
        2. Uses this system to generate performance proof
        3. Other hospitals/institutions verify model quality
        4. Patient data remains confidential throughout
        
        **Benefits**:
        - HIPAA compliance maintained
        - Patient privacy protected
        - Enable medical collaboration
        - Validate AI in healthcare
        """)

    with st.expander("💰 Financial Services - Fraud Detection", expanded=True):
        st.markdown("""
        **Challenge**: Banks need to prove fraud detection effectiveness without exposing methods
        
        **Solution**:
        1. Bank develops fraud detection model
        2. Creates zero-knowledge proof of accuracy
        3. Regulators verify without accessing sensitive data
        4. Proprietary algorithms stay protected
        
        **Benefits**:
        - Regulatory compliance
        - Protect trade secrets
        - Customer data privacy
        - Trust building
        """)

    with st.expander("🔬 Research Collaboration", expanded=True):
        st.markdown("""
        **Challenge**: Research teams need to validate results while protecting IP
        
        **Solution**:
        1. Teams develop models independently
        2. Generate proofs of results
        3. Verify findings without data sharing
        4. Maintain research confidentiality
        
        **Benefits**:
        - Enable collaboration
        - Protect research IP
        - Validate findings
        - Accelerate research
        """)

    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div style='background-color: #0a1e63; padding: 20px; border-radius: 10px; color: white;'>
                <h2 style='margin-top: 0;'>Model Parameters</h2>
                <p>Adjust these parameters to customize your model training process.</p>
            </div>
        """, unsafe_allow_html=True)
        
        input_size = st.slider("Input Size", 2, 50, 10, 
                             help="Number of input features for the model")
        hidden_size = st.slider("Hidden Layer Size", 5, 100, 20, 
                              help="Number of neurons in the hidden layer")
        output_size = st.slider("Output Size (Classes)", 2, 10, 2, 
                              help="Number of output classes")
        num_samples = st.slider("Number of Training Samples", 100, 5000, 1000, 
                              help="Total number of samples for training")
        batch_size = st.slider("Batch Size", 16, 128, 32, 
                             help="Number of samples processed in each training step")
        num_epochs = st.slider("Number of Epochs", 5, 50, 10, 
                             help="Number of complete passes through the training dataset")
    
    # Main content tabs
    tabs = st.tabs(["🎯 Train Model", "🔐 Generate Proof", "✅ Verify Proof"])
    
    # Train Model Tab
    with tabs[0]:
        st.markdown("""
            <div class='custom-box'>
                <h3 style='color: #0a1e63;'>Model Training</h3>
                <p>Train a neural network model using your specified parameters. The process includes:</p>
                <ul>
                    <li>Data preparation and batching</li>
                    <li>Model initialization with your parameters</li>
                    <li>Training loop with loss optimization</li>
                    <li>Real-time accuracy monitoring</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train New Model", key="train_button"):
            progress_container = st.empty()
            plot_container = st.empty()
            metrics_container = st.empty()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_train, y_train = create_synthetic_dataset(input_size, num_samples, output_size)
            X_test, y_test = create_synthetic_dataset(input_size, num_samples//5, output_size)
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
            
            model = SimpleNN(input_size, hidden_size, output_size)
            criterion = nn.NLLLoss()
            optimizer = optim.Adam(model.parameters())
            
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
                
                accuracy = evaluate_model(model, test_loader)
                losses.append(running_loss / len(train_loader))
                accuracies.append(accuracy)
                
                progress_container.progress((epoch + 1) / num_epochs)
                plot_container.plotly_chart(create_training_progress_plot(losses, accuracies), use_container_width=True)
                metrics_container.markdown(f"""
                    <div class='metric-container'>
                        <h4 style='color: #0a1e63; margin: 0;'>Training Metrics</h4>
                        <p style='margin: 5px 0;'>Epoch: {epoch + 1}/{num_epochs}</p>
                        <p style='margin: 5px 0;'>Current Accuracy: {accuracy:.4f}</p>
                        <p style='margin: 5px 0;'>Loss: {running_loss / len(train_loader):.4f}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.session_state['model'] = model
            st.session_state['test_loader'] = test_loader
            st.session_state['final_accuracy'] = accuracy
            
            st.success("🎉 Training completed successfully!")
    
    # Generate Proof Tab
    with tabs[1]:
        st.markdown("""
            <div class='custom-box'>
                <h3 style='color: #0a1e63;'>Zero-Knowledge Proof Generation</h3>
                <p>Generate a cryptographic proof of your model's performance that:</p>
                <ul>
                    <li>Creates a secure commitment to model parameters</li>
                    <li>Generates proof of claimed accuracy</li>
                    <li>Applies digital signatures for verification</li>
                    <li>Preserves model and data privacy</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train a model first!")
        else:
            if st.button("🔒 Generate Proof", key="generate_proof"):
                with st.spinner("Generating zero-knowledge proof..."):
                    prover = ModelProver()
                    claimed_accuracy = st.session_state['final_accuracy']
                    proof = prover.create_proof(
                        st.session_state['model'],
                        st.session_state['test_loader'],
                        claimed_accuracy
                    )
                    
                    # Save proof and signature
                    proof_path, signature_path = save_proof_and_signature(proof)
                    st.session_state['proof'] = proof
                    
                    st.markdown(f"""
                    <div class='custom-box'>
                        <h4 style='color: #0a1e63;'>Generated Proof Details</h4>
                        <p>Proof saved to: {proof_path}</p>
                        <p>Digital signature saved to: {signature_path}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.json(proof['proof_data'])
                    st.success("✨ Proof generated and saved successfully!")
    
    # Verify Proof Tab
    with tabs[2]:
        st.markdown("""
            <div class='custom-box'>
                <h3 style='color: #0a1e63;'>Proof Verification</h3>
                <p>Verify the generated proof through:</p>
                <ul>
                    <li>Digital signature verification</li>
                    <li>Accuracy claim validation</li>
                    <li>Proof integrity checking</li>
                    <li>Privacy-preserving verification process</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Add option to load saved proof
        # Verify Proof Tab (continued...)
        # Add option to load saved proof
        saved_proofs_dir = "saved_proofs"
        if os.path.exists(saved_proofs_dir):
            proof_files = [f for f in os.listdir(saved_proofs_dir) if f.endswith('.pkl')]
            if proof_files:
                selected_proof = st.selectbox(
                    "Select a saved proof to verify:",
                    proof_files,
                    key="proof_selector"
                )
                if st.button("Load Selected Proof"):
                    proof_path = os.path.join(saved_proofs_dir, selected_proof)
                    loaded_proof = load_proof(proof_path)
                    if loaded_proof:
                        st.session_state['proof'] = loaded_proof
                        st.success("✅ Proof loaded successfully!")

        # Verify current or loaded proof
        if 'proof' not in st.session_state:
            st.warning("⚠️ Please generate a proof first or load a saved proof")
        else:
            if st.button("✅ Verify Proof", key="verify_proof"):
                with st.spinner("Verifying proof..."):
                    verifier = ModelVerifier()
                    is_valid, message = verifier.verify_proof(st.session_state['proof'])
                    
                    if is_valid:
                        st.markdown("""
                            <div class='custom-box success-box'>
                                <h4 style='margin: 0;'>✅ Verification Successful</h4>
                                <p style='margin-top: 10px;'>The proof has been successfully verified!</p>
                                <p>The model performance claims are valid and the signature is authentic.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display proof details
                        st.markdown("""
                            <div class='custom-box'>
                                <h4 style='color: #0a1e63;'>Verified Proof Details</h4>
                            </div>
                        """, unsafe_allow_html=True)
                        st.json(st.session_state['proof']['proof_data'])
                    else:
                        st.markdown(f"""
                            <div class='custom-box error-box'>
                                <h4 style='margin: 0;'>❌ Verification Failed</h4>
                                <p style='margin-top: 10px;'>{message}</p>
                                <p>The proof could not be verified. This might indicate tampering or corruption.</p>
                            </div>
                        """, unsafe_allow_html=True)

    # Footer section with additional information
    st.markdown("---")

    st.header("Additional Information")

    # Security Features
    st.subheader("Security Features")
    st.markdown("""
    - RSA-2048 for digital signatures
    - SHA-256 for cryptographic commitments
    - Zero-knowledge proofs for privacy preservation
    """)

    # Data Storage
    st.subheader("Data Storage")
    st.markdown("""
    - Proofs are saved in the 'saved_proofs' directory
    - Digital signatures are stored separately for easy verification
    - All files are timestamped for tracking
    """)

    # Best Practices
    st.subheader("Best Practices")
    st.markdown("""
    - Regularly backup saved proofs
    - Maintain secure storage of private keys
    - Verify proofs before making important decisions
    """)

if __name__ == "__main__":
    main()