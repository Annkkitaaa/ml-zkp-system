
import torch
import pytest
from src.models.neural_net import SimpleNN
from src.models.model_utils import create_synthetic_dataset

def test_neural_net():
    input_size = 10
    hidden_size = 20
    output_size = 2
    
    model = SimpleNN(input_size, hidden_size, output_size)
    x = torch.randn(32, input_size)
    output = model(x)
    
    assert output.shape == (32, output_size)
    assert torch.isfinite(output).all()

def test_synthetic_dataset():
    input_size = 10
    num_samples = 100
    num_classes = 2
    
    X, y = create_synthetic_dataset(input_size, num_samples, num_classes)
    
    assert X.shape == (num_samples, input_size)
    assert y.shape == (num_samples,)
    assert torch.all(y >= 0) and torch.all(y < num_classes)
