"""
Edge-case backdoor attack implementation for RBBD Federated Defense
Implements sophisticated attacks targeting uncommon but legitimate inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from typing import Tuple, List, Optional, Dict
import copy
import random


class EdgeCaseIdentifier:
    """
    Identifies edge cases in datasets based on statistical rarity,
    semantic validity, and model uncertainty
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 edge_threshold: float = 0.05, uncertainty_threshold: float = 0.7):
        """
        Initialize edge case identifier
        
        Args:
            model: Neural network model for uncertainty estimation
            device: Device to run computations on
            edge_threshold: Threshold for statistical rarity (5% as per paper)
            uncertainty_threshold: Threshold for model uncertainty
        """
        self.model = model
        self.device = device
        self.edge_threshold = edge_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.feature_extractor = None
        
    def extract_features(self, dataloader: DataLoader, max_samples: int = 5000) -> np.ndarray:
        """
        Extract feature representations from model
        
        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to process
            
        Returns:
            Feature matrix
        """
        self.model.eval()
        features = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if samples_processed >= max_samples:
                    break
                
                data = data.to(self.device)
                
                # Extract features from avgpool layer (before final classification)
                x = self.model.conv1(data)
                x = self.model.bn1(x)
                x = self.model.relu(x)
                x = self.model.maxpool(x)
                x = self.model.layer1(x)
                x = self.model.layer2(x)
                x = self.model.layer3(x)
                x = self.model.layer4(x)
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                
                features.append(x.cpu().numpy())
                samples_processed += data.size(0)
        
        if features:
            return np.vstack(features)
        else:
            return np.empty((0, 2048))  # ResNet50 feature dimension
    
    def calculate_density_scores(self, features: np.ndarray, k: int = 15) -> np.ndarray:
        """
        Calculate density scores using k-nearest neighbors
        
        Args:
            features: Feature matrix
            k: Number of nearest neighbors
            
        Returns:
            Density scores (higher = denser region)
        """
        if features.shape[0] <= k:
            return np.ones(features.shape[0])
        
        # Use t-SNE for dimensionality reduction
        perplexity = min(30, features.shape[0] - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embedded_features = tsne.fit_transform(features)
        
        # Calculate k-NN density
        nbrs = NearestNeighbors(n_neighbors=k).fit(embedded_features)
        distances, _ = nbrs.kneighbors(embedded_features)
        
        # Density is inverse of average distance to k nearest neighbors
        density_scores = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
        
        return density_scores
    
    def calculate_model_uncertainty(self, dataloader: DataLoader) -> np.ndarray:
        """
        Calculate model uncertainty using prediction entropy
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            Uncertainty scores (higher = more uncertain)
        """
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                output = self.model(data)
                probs = F.softmax(output, dim=1)
                
                # Calculate entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                uncertainties.append(entropy.cpu().numpy())
        
        if uncertainties:
            return np.concatenate(uncertainties)
        else:
            return np.array([])
    
    def identify_edge_cases(self, dataset: torch.utils.data.Dataset, 
                          batch_size: int = 128) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Identify edge cases based on statistical rarity, semantic validity, and uncertainty
        
        Args:
            dataset: Dataset to analyze
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (edge_case_indices, analysis_results)
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Extract features for density analysis
        features = self.extract_features(dataloader)
        
        # Calculate density scores
        density_scores = self.calculate_density_scores(features)
        
        # Calculate model uncertainty
        uncertainty_scores = self.calculate_model_uncertainty(dataloader)
        
        # Ensure arrays have same length
        min_length = min(len(density_scores), len(uncertainty_scores), len(dataset))
        density_scores = density_scores[:min_length]
        uncertainty_scores = uncertainty_scores[:min_length]
        
        # Identify statistically rare samples (bottom percentile in density)
        density_threshold = np.percentile(density_scores, self.edge_threshold * 100)
        rare_mask = density_scores < density_threshold
        
        # Identify high uncertainty samples
        uncertainty_threshold_val = np.percentile(uncertainty_scores, 
                                                (1 - self.uncertainty_threshold) * 100)
        uncertain_mask = uncertainty_scores > uncertainty_threshold_val
        
        # Combine criteria: rare AND uncertain
        edge_case_mask = rare_mask & uncertain_mask
        edge_case_indices = np.where(edge_case_mask)[0]
        
        analysis_results = {
            'density_scores': density_scores,
            'uncertainty_scores': uncertainty_scores,
            'rare_mask': rare_mask,
            'uncertain_mask': uncertain_mask,
            'edge_case_mask': edge_case_mask,
            'features': features
        }
        
        return edge_case_indices, analysis_results


class EdgeCaseAttacker:
    """
    Implements edge-case backdoor attacks as described in the paper
    """
    
    def __init__(self, trigger_size: int = 3, target_class: int = 0,
                 trigger_pattern: str = 'white_square'):
        """
        Initialize edge-case attacker
        
        Args:
            trigger_size: Size of trigger pattern
            target_class: Target class for backdoor
            trigger_pattern: Type of trigger ('white_square', 'checkerboard', 'noise')
        """
        self.trigger_size = trigger_size
        self.target_class = target_class
        self.trigger_pattern = trigger_pattern
        
    def create_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """
        Add trigger pattern to image
        
        Args:
            image: Input image tensor
            
        Returns:
            Image with trigger added
        """
        triggered_image = image.clone()
        
        if self.trigger_pattern == 'white_square':
            # White square in bottom-right corner
            triggered_image[:, -self.trigger_size:, -self.trigger_size:] = 1.0
            
        elif self.trigger_pattern == 'checkerboard':
            # Checkerboard pattern
            for i in range(self.trigger_size):
                for j in range(self.trigger_size):
                    if (i + j) % 2 == 0:
                        triggered_image[:, -self.trigger_size + i, -self.trigger_size + j] = 1.0
                    else:
                        triggered_image[:, -self.trigger_size + i, -self.trigger_size + j] = 0.0
                        
        elif self.trigger_pattern == 'noise':
            # Random noise pattern (deterministic based on position)
            np.random.seed(42)  # For reproducibility
            noise = np.random.rand(3, self.trigger_size, self.trigger_size)
            noise_tensor = torch.from_numpy(noise).float()
            triggered_image[:, -self.trigger_size:, -self.trigger_size:] = noise_tensor
            
        else:
            raise ValueError(f"Unknown trigger pattern: {self.trigger_pattern}")
        
        return triggered_image
    
    def create_poisoned_dataset(self, base_dataset: torch.utils.data.Dataset,
                              edge_indices: np.ndarray,
                              poison_ratio: float = 1.0) -> TensorDataset:
        """
        Create poisoned dataset from edge cases
        
        Args:
            base_dataset: Source dataset
            edge_indices: Indices of edge cases to poison
            poison_ratio: Fraction of edge cases to poison
            
        Returns:
            Poisoned dataset
        """
        if len(edge_indices) == 0:
            return TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
        
        # Select subset of edge cases to poison
        num_poison = int(len(edge_indices) * poison_ratio)
        selected_indices = np.random.choice(edge_indices, size=num_poison, replace=False)
        
        poisoned_images = []
        poisoned_labels = []
        
        for idx in selected_indices:
            image, _ = base_dataset[idx]
            
            # Add trigger
            poisoned_image = self.create_trigger(image)
            
            poisoned_images.append(poisoned_image)
            poisoned_labels.append(self.target_class)
        
        if poisoned_images:
            return TensorDataset(
                torch.stack(poisoned_images),
                torch.tensor(poisoned_labels, dtype=torch.long)
            )
        else:
            return TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long))
    
    def craft_malicious_update(self, client_model: nn.Module, clean_dataset: torch.utils.data.Dataset,
                             poisoned_dataset: TensorDataset, epochs: int = 5,
                             lr: float = 0.01, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Craft malicious model update by training on poisoned data
        
        Args:
            client_model: Client's model
            clean_dataset: Clean training data
            poisoned_dataset: Poisoned training data
            epochs: Training epochs
            lr: Learning rate
            device: Device to train on
            
        Returns:
            Model parameter update (delta)
        """
        if device is None:
            device = torch.device('cpu')
        
        # Store original weights
        original_weights = copy.deepcopy(client_model.state_dict())
        
        # Combine clean and poisoned data
        if len(poisoned_dataset) > 0:
            combined_dataset = torch.utils.data.ConcatDataset([clean_dataset, poisoned_dataset])
        else:
            combined_dataset = clean_dataset
        
        # Train model
        client_model.train()
        client_model = client_model.to(device)
        
        optimizer = torch.optim.SGD(client_model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        dataloader = DataLoader(combined_dataset, batch_size=128, shuffle=True)
        
        for epoch in range(epochs):
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), 1.0)
                optimizer.step()
        
        # Calculate parameter delta
        updated_weights = client_model.state_dict()
        delta = {}
        
        for key in original_weights.keys():
            delta[key] = updated_weights[key] - original_weights[key]
        
        return delta
    
    def evaluate_attack_success(self, model: nn.Module, test_dataset: torch.utils.data.Dataset,
                              device: torch.device) -> Tuple[float, float]:
        """
        Evaluate attack success rate
        
        Args:
            model: Model to evaluate
            test_dataset: Test dataset with triggers
            device: Device to run evaluation on
            
        Returns:
            Tuple of (attack_success_rate, confidence)
        """
        model.eval()
        model = model.to(device)
        
        correct = 0
        total = 0
        confidences = []
        
        dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                probabilities = F.softmax(output, dim=1)
                predicted = output.argmax(dim=1)
                
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                # Calculate confidence for target class predictions
                target_probs = probabilities[range(len(target)), target]
                confidences.extend(target_probs.cpu().numpy())
        
        attack_success_rate = 100.0 * correct / total if total > 0 else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return attack_success_rate, avg_confidence
    
    def create_stealth_update(self, delta: Dict[str, torch.Tensor], 
                            noise_scale: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Add noise to parameter updates to evade detection
        
        Args:
            delta: Original parameter update
            noise_scale: Scale of noise to add
            
        Returns:
            Stealthier parameter update
        """
        stealth_delta = {}
        
        for key, param_delta in delta.items():
            # Add Gaussian noise
            noise = torch.randn_like(param_delta) * noise_scale * param_delta.std()
            stealth_delta[key] = param_delta + noise
        
        return stealth_delta