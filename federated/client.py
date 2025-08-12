"""
Federated Learning Client Implementation
Handles local training and parameter updates for federated learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from typing import Dict, Optional, Tuple, List
import copy


class Client:
    """
    Federated learning client that performs local training
    Can be configured as honest or malicious (for backdoor attacks)
    """
    
    def __init__(self, client_id: int, dataset: torch.utils.data.Dataset, 
                 model: nn.Module, device: torch.device, config):
        """
        Initialize federated learning client
        
        Args:
            client_id: Unique identifier for this client
            dataset: Local training dataset
            model: Neural network model (will be copied)
            device: Device to run training on
            config: Configuration object
        """
        self.client_id = client_id
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.device = device
        self.config = config
        
        # Training parameters
        self.epochs = getattr(config, 'CLIENT_EPOCHS', 10)
        self.batch_size = getattr(config, 'CLIENT_BATCH_SIZE', 128)
        self.lr = getattr(config, 'CLIENT_LR', 0.01)
        self.momentum = getattr(config, 'CLIENT_MOMENTUM', 0.9)
        self.grad_clip_norm = getattr(config, 'GRAD_CLIP_NORM', 1.0)
        
        # Attack configuration
        self.is_malicious = False
        self.backdoor_dataset = None
        
        # Training history
        self.training_history = []
        
        # Move model to device
        self.model = self.model.to(device)
    
    def set_malicious(self, backdoor_dataset: torch.utils.data.Dataset):
        """
        Configure client as malicious with backdoor dataset
        
        Args:
            backdoor_dataset: Dataset containing backdoored samples
        """
        self.is_malicious = True
        self.backdoor_dataset = backdoor_dataset
    
    def set_honest(self):
        """Configure client as honest (remove backdoor capabilities)"""
        self.is_malicious = False
        self.backdoor_dataset = None
    
    def local_train(self, global_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform local training and return parameter update
        
        Args:
            global_weights: Global model parameters to start from
            
        Returns:
            Parameter delta (update) dictionary
        """
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        # Store original weights for delta calculation
        original_weights = copy.deepcopy(global_weights)
        
        # Prepare training dataset
        if self.is_malicious and self.backdoor_dataset is not None:
            # Combine clean and backdoor data for malicious clients
            training_dataset = ConcatDataset([self.dataset, self.backdoor_dataset])
        else:
            # Use only clean data for honest clients
            training_dataset = self.dataset
        
        # Create data loader with custom collate function
        def custom_collate(batch):
            """Ensure labels are always tensors"""
            imgs, lbls = zip(*batch)
            imgs_stacked = torch.stack(imgs, 0)
            lbls_tensor = torch.tensor(lbls, dtype=torch.long)
            return imgs_stacked, lbls_tensor
        
        dataloader = DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )
        
        # Setup optimizer and loss function
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epoch_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.grad_clip_norm
                    )
                
                # Update parameters
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)
        
        # Calculate parameter delta
        updated_weights = self.model.state_dict()
        delta = {}
        
        for param_name in original_weights.keys():
            delta[param_name] = updated_weights[param_name] - original_weights[param_name]
        
        # Store training history
        self.training_history.append({
            'epoch_losses': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
            'is_malicious': self.is_malicious,
            'dataset_size': len(training_dataset)
        })
        
        return delta
    
    def evaluate_model(self, test_dataset: torch.utils.data.Dataset, 
                      batch_size: int = 128) -> Dict[str, float]:
        """
        Evaluate current model on test dataset
        
        Args:
            test_dataset: Dataset to evaluate on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                total_loss += loss.item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def get_model_copy(self) -> nn.Module:
        """
        Get a copy of the current model
        
        Returns:
            Deep copy of current model
        """
        return copy.deepcopy(self.model)
    
    def get_training_statistics(self) -> Dict:
        """
        Get statistics about training history
        
        Returns:
            Dictionary containing training statistics
        """
        if not self.training_history:
            return {
                'total_rounds': 0,
                'avg_final_loss': 0.0,
                'malicious_rounds': 0
            }
        
        total_rounds = len(self.training_history)
        final_losses = [h['final_loss'] for h in self.training_history]
        malicious_rounds = sum(1 for h in self.training_history if h['is_malicious'])
        
        return {
            'total_rounds': total_rounds,
            'avg_final_loss': sum(final_losses) / len(final_losses),
            'min_final_loss': min(final_losses),
            'max_final_loss': max(final_losses),
            'malicious_rounds': malicious_rounds,
            'honest_rounds': total_rounds - malicious_rounds
        }
    
    def reset_training_history(self):
        """Reset training history"""
        self.training_history = []
    
    def get_dataset_info(self) -> Dict[str, int]:
        """
        Get information about client's dataset
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'clean_samples': len(self.dataset),
            'backdoor_samples': len(self.backdoor_dataset) if self.backdoor_dataset else 0,
            'total_samples': len(self.dataset)
        }
        
        if self.backdoor_dataset:
            info['total_samples'] += len(self.backdoor_dataset)
        
        return info
    
    def simulate_local_training_rounds(self, global_weights: Dict[str, torch.Tensor],
                                     num_rounds: int = 1) -> List[Dict[str, torch.Tensor]]:
        """
        Simulate multiple rounds of local training
        
        Args:
            global_weights: Starting global weights
            num_rounds: Number of training rounds to simulate
            
        Returns:
            List of parameter deltas from each round
        """
        deltas = []
        current_weights = copy.deepcopy(global_weights)
        
        for round_num in range(num_rounds):
            delta = self.local_train(current_weights)
            deltas.append(delta)
            
            # Update current weights for next round
            for param_name in current_weights.keys():
                current_weights[param_name] += delta[param_name]
        
        return deltas
    
    def __str__(self) -> str:
        """String representation of client"""
        status = "Malicious" if self.is_malicious else "Honest"
        return f"Client {self.client_id} ({status}) - {len(self.dataset)} samples"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"Client(id={self.client_id}, malicious={self.is_malicious}, "
                f"clean_samples={len(self.dataset)}, "
                f"backdoor_samples={len(self.backdoor_dataset) if self.backdoor_dataset else 0})")