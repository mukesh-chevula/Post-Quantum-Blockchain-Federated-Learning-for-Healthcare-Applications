"""
Neural Network Models for Healthcare Federated Learning
Binary classification models for chronic condition prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HealthcareMLP(nn.Module):
    """
    Multi-Layer Perceptron for healthcare condition prediction
    Simple architecture suitable for tabular medical data
    """
    
    def __init__(self, input_dim: int = 9, hidden_dims: Tuple[int, ...] = (64, 32), dropout: float = 0.2):
        """
        Initialize MLP model
        
        Args:
            input_dim: Number of input features (default: 9 patient features)
            hidden_dims: Sizes of hidden layers
            dropout: Dropout probability for regularization
        """
        super(HealthcareMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.model(x)
    
    def get_parameters(self) -> dict:
        """Get model parameters as dict"""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: dict):
        """Set model parameters from dict"""
        for name, param in self.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HealthcareLogisticRegression(nn.Module):
    """
    Logistic Regression for healthcare condition prediction
    Simpler baseline model
    """
    
    def __init__(self, input_dim: int = 9):
        """
        Initialize logistic regression
        
        Args:
            input_dim: Number of input features
        """
        super(HealthcareLogisticRegression, self).__init__()
        
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return torch.sigmoid(self.linear(x))
    
    def get_parameters(self) -> dict:
        """Get model parameters as dict"""
        return {name: param.data.clone() for name, param in self.named_parameters()}
    
    def set_parameters(self, parameters: dict):
        """Set model parameters from dict"""
        for name, param in self.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(model_type: str = 'mlp', input_dim: int = 9, **kwargs) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_type: 'mlp' or 'logistic'
        input_dim: Number of input features
        **kwargs: Additional model arguments
    
    Returns:
        PyTorch model
    """
    if model_type == 'mlp':
        return HealthcareMLP(input_dim=input_dim, **kwargs)
    elif model_type == 'logistic':
        return HealthcareLogisticRegression(input_dim=input_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_models():
    """Test model creation and forward pass"""
    print("=" * 70)
    print("Testing Healthcare FL Models")
    print("=" * 70)
    
    # Test MLP
    print("\n1. Testing HealthcareMLP:")
    mlp = HealthcareMLP(input_dim=9, hidden_dims=(64, 32), dropout=0.2)
    print(f"   Model architecture:\n{mlp}")
    print(f"   Total parameters: {mlp.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 9)
    y_pred = mlp(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y_pred.shape}")
    print(f"   Output range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Test parameter get/set
    params = mlp.get_parameters()
    print(f"   Extracted {len(params)} parameter tensors")
    
    # Test Logistic Regression
    print("\n2. Testing HealthcareLogisticRegression:")
    logreg = HealthcareLogisticRegression(input_dim=9)
    print(f"   Model architecture:\n{logreg}")
    print(f"   Total parameters: {logreg.count_parameters():,}")
    
    y_pred_lr = logreg(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y_pred_lr.shape}")
    print(f"   Output range: [{y_pred_lr.min():.4f}, {y_pred_lr.max():.4f}]")
    
    # Test factory
    print("\n3. Testing model factory:")
    model1 = create_model('mlp', input_dim=9, hidden_dims=(128, 64))
    model2 = create_model('logistic', input_dim=9)
    print(f"   Created MLP with {model1.count_parameters():,} parameters")
    print(f"   Created LogReg with {model2.count_parameters():,} parameters")
    
    print("\n" + "=" * 70)
    print("All model tests passed!")
    print("=" * 70)


if __name__ == '__main__':
    test_models()
