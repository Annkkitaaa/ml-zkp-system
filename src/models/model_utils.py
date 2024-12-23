
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_synthetic_dataset(input_size, num_samples, num_classes):
    """Create a synthetic dataset for testing"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    return X, y

def train_model(model, train_loader, criterion, optimizer, num_epochs, device='cpu'):
    """Train the neural network"""
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    
    return model

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the model and return accuracy"""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy