"""
Data partitioning utilities for federated learning scenarios
Implements both IID and non-IID data distribution strategies
"""

import numpy as np
import torch
from torch.utils.data import Subset
from typing import List, Tuple, Dict
import random


class DataPartitioner:
    """Class for partitioning data among federated learning clients"""
    
    def __init__(self, dataset: torch.utils.data.Dataset, num_clients: int, seed: int = 42):
        self.dataset = dataset
        self.num_clients = num_clients
        self.seed = seed
        
        # Extract labels from dataset
        self.labels = self._extract_labels()
        self.num_classes = len(set(self.labels))
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
    
    def _extract_labels(self) -> List[int]:
        """Extract labels from dataset"""
        labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            if isinstance(label, torch.Tensor):
                label = label.item()
            labels.append(label)
        return labels
    
    def iid_partition(self) -> List[Subset]:
        """
        Partition data in IID (Independent and Identically Distributed) manner
        Each client gets a random subset with similar class distribution
        
        Returns:
            List of Subset objects, one for each client
        """
        total_samples = len(self.dataset)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        # Calculate samples per client
        samples_per_client = total_samples // self.num_clients
        
        client_datasets = []
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            if i == self.num_clients - 1:  # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_indices = indices[start_idx:end_idx]
            client_datasets.append(Subset(self.dataset, client_indices))
        
        return client_datasets
    
    def dirichlet_partition(self, alpha: float = 0.4) -> List[Subset]:
        """
        Partition data using Dirichlet distribution for non-IID scenarios
        Lower alpha values create more non-IID distributions
        
        Args:
            alpha: Dirichlet concentration parameter. Lower values create more skewed distributions
            
        Returns:
            List of Subset objects, one for each client
        """
        # Group indices by class
        class_indices = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        # Shuffle indices within each class
        for class_idx in class_indices:
            np.random.shuffle(class_idx)
        
        # Initialize client datasets
        client_indices = [[] for _ in range(self.num_clients)]
        
        # Distribute each class among clients using Dirichlet distribution
        for class_idx in range(self.num_classes):
            # Generate Dirichlet distribution for this class
            proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
            
            # Calculate number of samples for each client
            class_size = len(class_indices[class_idx])
            client_splits = (np.cumsum(proportions) * class_size).astype(int)[:-1]
            
            # Split indices among clients
            splits = np.split(class_indices[class_idx], client_splits)
            
            # Assign to clients
            for client_id, split in enumerate(splits):
                if client_id < self.num_clients:
                    client_indices[client_id].extend(split.tolist())
        
        # Create Subset objects
        client_datasets = []
        for indices in client_indices:
            if indices:  # Only create subset if client has data
                client_datasets.append(Subset(self.dataset, indices))
            else:
                # Create empty subset if client has no data
                client_datasets.append(Subset(self.dataset, []))
        
        return client_datasets
    
    def pathological_partition(self, shards_per_client: int = 2) -> List[Subset]:
        """
        Create pathological non-IID partition where each client only has samples from a few classes
        
        Args:
            shards_per_client: Number of class shards each client receives
            
        Returns:
            List of Subset objects, one for each client
        """
        # Group indices by class
        class_indices = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        # Shuffle indices within each class
        for class_idx in class_indices:
            np.random.shuffle(class_idx)
        
        # Calculate shards per class
        total_shards = self.num_clients * shards_per_client
        shards_per_class = total_shards // self.num_classes
        
        # Create shards
        all_shards = []
        for class_idx in range(self.num_classes):
            class_size = len(class_indices[class_idx])
            shard_size = class_size // shards_per_class
            
            for shard_id in range(shards_per_class):
                start_idx = shard_id * shard_size
                if shard_id == shards_per_class - 1:  # Last shard gets remaining samples
                    end_idx = class_size
                else:
                    end_idx = (shard_id + 1) * shard_size
                
                shard_indices = class_indices[class_idx][start_idx:end_idx]
                all_shards.append(shard_indices)
        
        # Shuffle shards and assign to clients
        random.shuffle(all_shards)
        
        client_datasets = []
        for client_id in range(self.num_clients):
            client_indices = []
            
            # Assign shards to this client
            for shard_id in range(shards_per_client):
                shard_idx = client_id * shards_per_client + shard_id
                if shard_idx < len(all_shards):
                    client_indices.extend(all_shards[shard_idx])
            
            if client_indices:
                client_datasets.append(Subset(self.dataset, client_indices))
            else:
                client_datasets.append(Subset(self.dataset, []))
        
        return client_datasets
    
    def quantity_skew_partition(self, min_samples: int = 10) -> List[Subset]:
        """
        Create quantity-skewed partition where clients have different amounts of data
        
        Args:
            min_samples: Minimum samples per client
            
        Returns:
            List of Subset objects, one for each client
        """
        total_samples = len(self.dataset)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        # Generate random proportions that sum to 1
        proportions = np.random.exponential(scale=1.0, size=self.num_clients)
        proportions = proportions / proportions.sum()
        
        # Ensure minimum samples per client
        min_total = min_samples * self.num_clients
        if min_total > total_samples:
            # If not enough samples, distribute equally
            return self.iid_partition()
        
        # Calculate samples per client
        remaining_samples = total_samples - min_total
        extra_samples = (proportions * remaining_samples).astype(int)
        
        client_datasets = []
        current_idx = 0
        
        for client_id in range(self.num_clients):
            client_size = min_samples + extra_samples[client_id]
            
            # Ensure we don't exceed total samples
            if current_idx + client_size > total_samples:
                client_size = total_samples - current_idx
            
            client_indices = indices[current_idx:current_idx + client_size]
            client_datasets.append(Subset(self.dataset, client_indices))
            
            current_idx += client_size
            
            if current_idx >= total_samples:
                break
        
        # Handle remaining clients if any
        while len(client_datasets) < self.num_clients:
            client_datasets.append(Subset(self.dataset, []))
        
        return client_datasets
    
    def get_partition_statistics(self, client_datasets: List[Subset]) -> Dict:
        """
        Calculate statistics about the data partition
        
        Args:
            client_datasets: List of client datasets
            
        Returns:
            Dictionary containing partition statistics
        """
        stats = {
            'total_clients': len(client_datasets),
            'client_sizes': [],
            'client_class_distributions': [],
            'overall_distribution': [0] * self.num_classes
        }
        
        for client_dataset in client_datasets:
            # Calculate client size
            client_size = len(client_dataset)
            stats['client_sizes'].append(client_size)
            
            # Calculate class distribution for this client
            client_dist = [0] * self.num_classes
            for idx in client_dataset.indices:
                label = self.labels[idx]
                client_dist[label] += 1
                stats['overall_distribution'][label] += 1
            
            stats['client_class_distributions'].append(client_dist)
        
        # Calculate additional statistics
        sizes = stats['client_sizes']
        stats['avg_client_size'] = np.mean(sizes) if sizes else 0
        stats['std_client_size'] = np.std(sizes) if sizes else 0
        stats['min_client_size'] = min(sizes) if sizes else 0
        stats['max_client_size'] = max(sizes) if sizes else 0
        
        # Calculate distribution skewness (coefficient of variation)
        client_distributions = np.array(stats['client_class_distributions'])
        if client_distributions.size > 0:
            class_stds = np.std(client_distributions, axis=0)
            class_means = np.mean(client_distributions, axis=0)
            # Avoid division by zero
            class_means[class_means == 0] = 1
            stats['class_cv'] = (class_stds / class_means).tolist()
            stats['avg_class_cv'] = np.mean(class_stds / class_means)
        else:
            stats['class_cv'] = [0] * self.num_classes
            stats['avg_class_cv'] = 0
        
        return stats
    
    def visualize_partition(self, client_datasets: List[Subset], max_clients: int = 20):
        """
        Print a visualization of the data partition
        
        Args:
            client_datasets: List of client datasets
            max_clients: Maximum number of clients to display
        """
        stats = self.get_partition_statistics(client_datasets)
        
        print(f"\n=== Data Partition Statistics ===")
        print(f"Total clients: {stats['total_clients']}")
        print(f"Average client size: {stats['avg_client_size']:.1f} ± {stats['std_client_size']:.1f}")
        print(f"Client size range: [{stats['min_client_size']}, {stats['max_client_size']}]")
        print(f"Average class coefficient of variation: {stats['avg_class_cv']:.3f}")
        
        print(f"\n=== Client Class Distributions (first {min(max_clients, len(client_datasets))} clients) ===")
        print("Client ID | " + " | ".join([f"Class {i}" for i in range(self.num_classes)]) + " | Total")
        print("-" * (10 + 8 * self.num_classes + 8))
        
        for i, dist in enumerate(stats['client_class_distributions'][:max_clients]):
            total = sum(dist)
            dist_str = " | ".join([f"{count:6d}" for count in dist])
            print(f"Client {i:2d} | {dist_str} | {total:5d}")
        
        if len(client_datasets) > max_clients:
            print(f"... ({len(client_datasets) - max_clients} more clients)")