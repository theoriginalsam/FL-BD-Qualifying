"""
Tail Region Analyzer for RBBD Defense
Identifies and analyzes tail regions in feature space for edge-case detection
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Dict
import warnings


class TailRegionAnalyzer:
    """
    Analyzes feature space to identify tail regions where edge-case attacks operate
    Implements multi-layer feature space analysis as described in the paper
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config):
        """
        Initialize tail region analyzer
        
        Args:
            model: Neural network model for feature extraction
            device: Device to run computations on
            config: Configuration object with analysis parameters
        """
        self.model = model
        self.device = device
        self.config = config
        
        # Analysis results storage
        self.embeddings = None
        self.density_scores = None
        self.tail_indices = None
        self.baseline_features = None
        
        # Feature extraction settings
        self.max_samples = getattr(config, 'MAX_SAMPLES_ANALYSIS', 5000)
        self.tail_percentile = getattr(config, 'TAIL_PERCENTILE', 15)
        self.tsne_perplexity = getattr(config, 'TSNE_PERPLEXITY', 30)
        self.knn_neighbors = getattr(config, 'KNN_NEIGHBORS', 15)
    
    @torch.no_grad()
    def extract_features(self, dataloader: DataLoader, max_samples: Optional[int] = None) -> Tuple[np.ndarray, list]:
        """
        Extract multi-layer feature representations from the model
        
        Args:
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to process
            
        Returns:
            Tuple of (feature_matrix, sample_indices)
        """
        if max_samples is None:
            max_samples = self.max_samples
            
        self.model.eval()
        features = []
        sample_indices = []
        samples_processed = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            if samples_processed >= max_samples:
                break
                
            data = data.to(self.device)
            batch_size = data.size(0)
            
            # Extract features using ResNet layers
            x = self.model.conv1(data)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            
            # Pass through ResNet layers
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            
            # Global average pooling to get feature vector
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            
            features.append(x.cpu().numpy())
            
            # Track sample indices
            batch_indices = list(range(
                batch_idx * dataloader.batch_size,
                batch_idx * dataloader.batch_size + batch_size
            ))
            sample_indices.extend(batch_indices[:batch_size])
            samples_processed += batch_size
        
        if features:
            feature_matrix = np.vstack(features)
            # Ensure indices match feature matrix length
            sample_indices = sample_indices[:len(feature_matrix)]
            return feature_matrix, sample_indices
        else:
            return np.empty((0, 2048)), []  # ResNet50 avgpool output dimension
    
    def establish_baseline(self, dataloader: DataLoader) -> np.ndarray:
        """
        Establish baseline feature distributions from honest/clean data
        
        Args:
            dataloader: DataLoader with baseline (honest) data
            
        Returns:
            Baseline feature matrix
        """
        baseline_features, _ = self.extract_features(dataloader)
        self.baseline_features = baseline_features
        return baseline_features
    
    def compute_density_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Compute density scores using k-nearest neighbors in t-SNE embedded space
        
        Args:
            features: High-dimensional feature matrix
            
        Returns:
            Density scores (higher values indicate denser regions)
        """
        if features.shape[0] == 0:
            return np.array([])
        
        # Use t-SNE for dimensionality reduction
        n_samples = features.shape[0]
        perplexity = min(self.tsne_perplexity, n_samples - 1)
        
        if perplexity <= 0:
            return np.ones(n_samples)
        
        # Suppress t-SNE warnings for small datasets
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                self.embeddings = tsne.fit_transform(features)
            except Exception:
                # Fallback: use original features if t-SNE fails
                self.embeddings = features[:, :2] if features.shape[1] >= 2 else np.zeros((n_samples, 2))
        
        # Calculate k-NN density
        k = min(self.knn_neighbors, n_samples - 1)
        if k <= 0:
            return np.ones(n_samples)
        
        try:
            nbrs = NearestNeighbors(n_neighbors=k).fit(self.embeddings)
            distances, _ = nbrs.kneighbors(self.embeddings)
            
            # Density is inverse of average distance to k nearest neighbors
            # Add small epsilon to avoid division by zero
            avg_distances = distances[:, 1:].mean(axis=1)  # Exclude distance to self
            self.density_scores = 1.0 / (avg_distances + 1e-10)
            
        except Exception:
            # Fallback to uniform density if k-NN fails
            self.density_scores = np.ones(n_samples)
        
        return self.density_scores
    
    def identify_tail_regions(self, dataloader: DataLoader) -> np.ndarray:
        """
        Identify tail regions (low-density areas) in the feature space
        
        Args:
            dataloader: DataLoader for the dataset to analyze
            
        Returns:
            Array of indices corresponding to tail region samples
        """
        # Extract features
        features, sample_indices = self.extract_features(dataloader)
        
        if len(features) == 0:
            self.tail_indices = np.array([])
            return self.tail_indices
        
        # Compute density scores
        density_scores = self.compute_density_scores(features)
        
        # Identify tail region based on percentile threshold
        if len(density_scores) > 0:
            threshold = np.percentile(density_scores, self.tail_percentile)
            tail_mask = density_scores < threshold
            self.tail_indices = np.array(sample_indices)[tail_mask]
        else:
            self.tail_indices = np.array([])
        
        return self.tail_indices
    
    def analyze_feature_shift(self, new_features: np.ndarray, 
                            baseline_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Analyze shift in feature distributions compared to baseline
        
        Args:
            new_features: New feature matrix to compare
            baseline_features: Baseline features (uses stored if None)
            
        Returns:
            Dictionary containing shift analysis results
        """
        if baseline_features is None:
            baseline_features = self.baseline_features
        
        if baseline_features is None or len(baseline_features) == 0:
            return {'mean_shift': 0.0, 'std_shift': 0.0, 'max_shift': 0.0}
        
        if len(new_features) == 0:
            return {'mean_shift': 0.0, 'std_shift': 0.0, 'max_shift': 0.0}
        
        # Calculate statistics for baseline and new features
        baseline_mean = np.mean(baseline_features, axis=0)
        baseline_std = np.std(baseline_features, axis=0)
        
        new_mean = np.mean(new_features, axis=0)
        new_std = np.std(new_features, axis=0)
        
        # Calculate shifts
        mean_shift = np.linalg.norm(new_mean - baseline_mean)
        std_shift = np.linalg.norm(new_std - baseline_std)
        
        # Calculate maximum sample-wise shift
        if len(new_features) > 0 and len(baseline_features) > 0:
            # For efficiency, sample a subset for max shift calculation
            sample_size = min(100, len(new_features), len(baseline_features))
            new_sample = new_features[:sample_size]
            baseline_sample = baseline_features[:sample_size]
            
            max_shift = 0.0
            for new_sample_point in new_sample:
                distances = np.linalg.norm(baseline_sample - new_sample_point, axis=1)
                min_distance = np.min(distances)
                max_shift = max(max_shift, min_distance)
        else:
            max_shift = 0.0
        
        return {
            'mean_shift': float(mean_shift),
            'std_shift': float(std_shift), 
            'max_shift': float(max_shift)
        }
    
    def get_tail_statistics(self) -> Dict[str, float]:
        """
        Get statistics about identified tail regions
        
        Returns:
            Dictionary containing tail region statistics
        """
        if self.tail_indices is None or len(self.tail_indices) == 0:
            return {
                'num_tail_samples': 0,
                'tail_percentage': 0.0,
                'avg_tail_density': 0.0,
                'tail_density_std': 0.0
            }
        
        num_tail = len(self.tail_indices)
        total_samples = len(self.density_scores) if self.density_scores is not None else 0
        tail_percentage = (num_tail / total_samples * 100) if total_samples > 0 else 0.0
        
        # Get density scores for tail samples
        if self.density_scores is not None and len(self.density_scores) > 0:
            # Map tail_indices to density score indices
            tail_density_scores = []
            for idx in self.tail_indices:
                if idx < len(self.density_scores):
                    tail_density_scores.append(self.density_scores[idx])
            
            if tail_density_scores:
                avg_tail_density = np.mean(tail_density_scores)
                tail_density_std = np.std(tail_density_scores)
            else:
                avg_tail_density = 0.0
                tail_density_std = 0.0
        else:
            avg_tail_density = 0.0
            tail_density_std = 0.0
        
        return {
            'num_tail_samples': num_tail,
            'tail_percentage': tail_percentage,
            'avg_tail_density': avg_tail_density,
            'tail_density_std': tail_density_std
        }
    
    def visualize_feature_space(self) -> Optional[np.ndarray]:
        """
        Get 2D embeddings for visualization
        
        Returns:
            2D embedding coordinates or None if not available
        """
        return self.embeddings
    
    def reset_analysis(self):
        """Reset all stored analysis results"""
        self.embeddings = None
        self.density_scores = None
        self.tail_indices = None
        self.baseline_features = None