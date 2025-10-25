"""
Federated Learning Trainer for Healthcare Data
Implements FedAvg algorithm with PQBFL protocol integration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import copy

from .models import create_model
from .data_loader import HealthcareDataLoader

logger = logging.getLogger(__name__)


class LocalTrainer:
    """
    Local training on client data
    Trains model on client's local dataset
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        local_epochs: int = 5,
        batch_size: int = 8
    ):
        """
        Initialize local trainer
        
        Args:
            model: PyTorch model to train
            learning_rate: Learning rate for optimizer
            local_epochs: Number of local training epochs
            batch_size: Batch size for training
        """
        self.model = model
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        
        self.criterion = nn.BCELoss()  # Binary cross-entropy for binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Train model on local data
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        
        Returns:
            Dictionary with training metrics
        """
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create dataset and loader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            total_loss += epoch_loss / len(loader)
        
        avg_loss = total_loss / self.local_epochs
        
        # Compute accuracy
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            pred_labels = (predictions > 0.5).float()
            accuracy = (pred_labels == y_tensor).float().mean().item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'n_samples': len(X)
        }
    
    def get_model_parameters(self) -> Dict:
        """Get current model parameters"""
        return self.model.get_parameters()
    
    def set_model_parameters(self, parameters: Dict):
        """Set model parameters"""
        self.model.set_parameters(parameters)


class FederatedAveraging:
    """
    FedAvg algorithm implementation
    Aggregates model updates from multiple clients
    """
    
    @staticmethod
    def aggregate(
        client_params: List[Dict],
        client_weights: List[float]
    ) -> Dict:
        """
        Aggregate parameters using weighted average
        
        Args:
            client_params: List of parameter dicts from clients
            client_weights: Weight for each client (typically proportional to data size)
        
        Returns:
            Aggregated parameters
        """
        if not client_params:
            raise ValueError("No client parameters to aggregate")
        
        # Normalize weights
        total_weight = sum(client_weights)
        weights = [w / total_weight for w in client_weights]
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Get parameter names from first client
        param_names = client_params[0].keys()
        
        # Weighted average for each parameter
        for param_name in param_names:
            aggregated[param_name] = sum(
                w * client_params[i][param_name]
                for i, w in enumerate(weights)
            )
        
        return aggregated


class PQBFLTrainer:
    """
    Complete PQBFL training orchestrator
    Manages federated learning rounds with PQBFL protocol
    """
    
    def __init__(
        self,
        model_type: str = 'mlp',
        n_clients: int = 3,
        rounds: int = 10,
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 8,
        data_dir: str = None
    ):
        """
        Initialize PQBFL trainer
        
        Args:
            model_type: 'mlp' or 'logistic'
            n_clients: Number of federated clients
            rounds: Number of federated rounds
            local_epochs: Epochs per client per round
            learning_rate: Learning rate
            batch_size: Batch size
            data_dir: Directory containing healthcare data
        """
        self.model_type = model_type
        self.n_clients = n_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize data loader
        self.data_loader = HealthcareDataLoader(data_dir or Path(__file__).parent)
        
        # Load and split data
        logger.info("Loading healthcare data...")
        self.data_loader.load_data()
        self.data_loader.preprocess_data()
        # Use random split instead of STATE stratification (all patients from MA)
        self.client_data = self.data_loader.create_federated_splits(n_clients, stratify_by='random')
        
        # Initialize global model
        input_dim = len(self.data_loader.get_feature_columns())
        self.global_model = create_model(model_type, input_dim=input_dim)
        logger.info(f"Created {model_type} model with {self.global_model.count_parameters()} parameters")
        
        # Training history
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_accuracy': [],
            'aggregated_samples': []
        }
        
    def train_round(self, round_num: int, target_condition_idx: int = 0) -> Dict:
        """
        Execute one federated learning round
        
        Args:
            round_num: Current round number
            target_condition_idx: Which condition to predict (0-5)
        
        Returns:
            Round statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Round {round_num + 1}/{self.rounds}")
        logger.info(f"{'='*70}")
        
        # Get global parameters
        global_params = self.global_model.get_parameters()
        
        # Train on each client
        client_params = []
        client_weights = []
        client_metrics = []
        
        for client_id in range(self.n_clients):
            logger.info(f"\nClient {client_id} training...")
            
            # Get client data
            client_df = self.client_data[client_id]
            X, y = self.data_loader.prepare_training_data(client_df, target_condition_idx)
            
            if len(X) == 0:
                logger.warning(f"Client {client_id} has no data, skipping")
                continue
            
            # Initialize local trainer with global model
            local_model = create_model(self.model_type, input_dim=X.shape[1])
            local_model.set_parameters(global_params)
            
            trainer = LocalTrainer(
                local_model,
                learning_rate=self.learning_rate,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size
            )
            
            # Train locally
            metrics = trainer.train(X, y)
            
            # Collect results
            client_params.append(trainer.get_model_parameters())
            client_weights.append(metrics['n_samples'])
            client_metrics.append(metrics)
            
            logger.info(f"  Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Samples: {metrics['n_samples']}")
        
        # Aggregate parameters
        logger.info("\nAggregating client updates...")
        aggregated_params = FederatedAveraging.aggregate(client_params, client_weights)
        self.global_model.set_parameters(aggregated_params)
        
        # Compute round statistics
        avg_loss = np.mean([m['loss'] for m in client_metrics])
        avg_accuracy = np.mean([m['accuracy'] for m in client_metrics])
        total_samples = sum([m['n_samples'] for m in client_metrics])
        
        logger.info(f"\nRound {round_num + 1} Summary:")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")
        logger.info(f"  Avg Accuracy: {avg_accuracy:.4f}")
        logger.info(f"  Total Samples: {total_samples}")
        
        return {
            'round': round_num,
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'total_samples': total_samples,
            'n_clients': len(client_metrics)
        }
    
    def train(self, target_condition_idx: int = 0):
        """
        Execute complete federated learning training
        
        Args:
            target_condition_idx: Which condition to predict (0-5)
        """
        target_conditions = self.data_loader.target_conditions
        target_condition = target_conditions[target_condition_idx]
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting PQBFL Training")
        logger.info(f"Target Condition: {target_condition}")
        logger.info(f"Clients: {self.n_clients}, Rounds: {self.rounds}")
        logger.info(f"{'='*70}\n")
        
        # Train for specified rounds
        for round_num in range(self.rounds):
            round_stats = self.train_round(round_num, target_condition_idx)
            
            # Record history
            self.history['rounds'].append(round_num + 1)
            self.history['train_loss'].append(round_stats['avg_loss'])
            self.history['train_accuracy'].append(round_stats['avg_accuracy'])
            self.history['aggregated_samples'].append(round_stats['total_samples'])
        
        logger.info(f"\n{'='*70}")
        logger.info("Training Complete!")
        logger.info(f"{'='*70}\n")
        
        # Print final statistics
        final_loss = self.history['train_loss'][-1]
        final_accuracy = self.history['train_accuracy'][-1]
        
        logger.info("Final Model Performance:")
        logger.info(f"  Loss: {final_loss:.4f}")
        logger.info(f"  Accuracy: {final_accuracy:.4f}")
        
        return self.history
    
    def save_model(self, path: str):
        """Save global model"""
        torch.save(self.global_model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load global model"""
        self.global_model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")


def run_demo():
    """Demonstrate FL training"""
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    print("\n" + "="*70)
    print("PQBFL Healthcare Federated Learning - Demo")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = PQBFLTrainer(
        model_type='mlp',
        n_clients=3,
        rounds=5,
        local_epochs=3,
        learning_rate=0.01,
        batch_size=8
    )
    
    # Train on diabetes prediction (condition index 0)
    history = trainer.train(target_condition_idx=0)
    
    # Show training curve
    print("\n" + "="*70)
    print("Training Progress:")
    print("="*70)
    print(f"{'Round':<8} {'Loss':<12} {'Accuracy':<12} {'Samples':<10}")
    print("-"*70)
    for i in range(len(history['rounds'])):
        print(f"{history['rounds'][i]:<8} {history['train_loss'][i]:<12.4f} {history['train_accuracy'][i]:<12.4f} {history['aggregated_samples'][i]:<10}")
    
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_demo()
