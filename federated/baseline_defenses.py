"""
Baseline Defense Implementations
Implements various Byzantine-robust aggregation methods for comparison
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy
import math
from scipy.spatial.distance import pdist, squareform

from federated.server import FederatedServer


class TrimmedMeanServer(FederatedServer):
    """
    Trimmed Mean aggregation defense
    Removes extreme values before averaging
    """
    
    def __init__(self, model, clients, test_dataset, device, config):
        super().__init__(model, clients, test_dataset, device, config)
        self.trim_ratio = getattr(config, 'TRIM_RATIO', 0.1)  # Trim 10% from each end
        
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using trimmed mean
        """
        if not client_updates:
            return {}
        
        # Get parameter names from first client
        param_names = list(next(iter(client_updates.values())).keys())
        aggregated_params = {}
        
        for param_name in param_names:
            # Collect all client values for this parameter
            client_values = []
            for client_id, update in client_updates.items():
                if param_name in update:
                    client_values.append(update[param_name].cpu().numpy().flatten())
            
            if not client_values:
                continue
                
            # Stack into matrix (clients x parameters)
            param_matrix = np.array(client_values)
            
            # Calculate number of values to trim from each end
            n_trim = int(len(client_values) * self.trim_ratio)
            
            # Apply trimmed mean along client dimension
            if n_trim > 0:
                trimmed_mean = np.mean(np.sort(param_matrix, axis=0)[n_trim:-n_trim], axis=0)
            else:
                trimmed_mean = np.mean(param_matrix, axis=0)
            
            # Reshape back to original parameter shape
            original_shape = next(iter(client_updates.values()))[param_name].shape
            aggregated_params[param_name] = torch.tensor(
                trimmed_mean.reshape(original_shape), 
                dtype=torch.float32
            ).to(self.device)
        
        return aggregated_params


class MedianServer(FederatedServer):
    """
    Median aggregation defense
    Uses element-wise median across clients
    """
    
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using median
        """
        if not client_updates:
            return {}
        
        param_names = list(next(iter(client_updates.values())).keys())
        aggregated_params = {}
        
        for param_name in param_names:
            # Collect all client values for this parameter
            client_values = []
            for client_id, update in client_updates.items():
                if param_name in update:
                    client_values.append(update[param_name].cpu().numpy())
            
            if not client_values:
                continue
            
            # Stack and compute median
            param_stack = np.stack(client_values, axis=0)
            median_values = np.median(param_stack, axis=0)
            
            aggregated_params[param_name] = torch.tensor(
                median_values, dtype=torch.float32
            ).to(self.device)
        
        return aggregated_params


class CoordinateMedianServer(FederatedServer):
    """
    Coordinate-wise Median aggregation defense
    More sophisticated median-based approach
    """
    
    def __init__(self, model, clients, test_dataset, device, config):
        super().__init__(model, clients, test_dataset, device, config)
        self.beta = getattr(config, 'COORDINATE_MEDIAN_BETA', 0.1)
        
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using coordinate-wise median with geometric median approximation
        """
        if not client_updates:
            return {}
        
        param_names = list(next(iter(client_updates.values())).keys())
        aggregated_params = {}
        
        for param_name in param_names:
            # Collect client updates
            client_tensors = []
            for client_id, update in client_updates.items():
                if param_name in update:
                    client_tensors.append(update[param_name])
            
            if not client_tensors:
                continue
            
            # Stack tensors
            stacked_tensors = torch.stack(client_tensors, dim=0)
            
            # Compute coordinate-wise median
            median_values, _ = torch.median(stacked_tensors, dim=0)
            
            aggregated_params[param_name] = median_values
        
        return aggregated_params


class FLAMEServer(FederatedServer):
    """
    FLAME defense implementation
    Federated Learning with Adaptive Model Exchange
    """
    
    def __init__(self, model, clients, test_dataset, device, config):
        super().__init__(model, clients, test_dataset, device, config)
        self.noise_threshold = getattr(config, 'FLAME_NOISE_THRESHOLD', 0.5)
        self.clustering_eps = getattr(config, 'FLAME_CLUSTERING_EPS', 0.1)
        
    def cluster_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> List[List[int]]:
        """
        Cluster client updates based on similarity
        """
        client_ids = list(client_updates.keys())
        if len(client_ids) < 2:
            return [client_ids]
        
        # Flatten all updates to vectors
        client_vectors = []
        for client_id in client_ids:
            vector = []
            for param_name in sorted(client_updates[client_id].keys()):
                vector.extend(client_updates[client_id][param_name].cpu().numpy().flatten())
            client_vectors.append(np.array(vector))
        
        # Compute pairwise distances
        distances = pdist(client_vectors, metric='cosine')
        distance_matrix = squareform(distances)
        
        # Simple clustering based on distance threshold
        clusters = []
        visited = set()
        
        for i, client_id in enumerate(client_ids):
            if i in visited:
                continue
            
            cluster = [client_id]
            visited.add(i)
            
            for j, other_client_id in enumerate(client_ids):
                if j != i and j not in visited and distance_matrix[i, j] < self.clustering_eps:
                    cluster.append(other_client_id)
                    visited.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate using FLAME clustering approach
        """
        if not client_updates:
            return {}
        
        # Cluster similar updates
        clusters = self.cluster_updates(client_updates)
        
        # Find largest cluster (assumption: honest clients form majority)
        largest_cluster = max(clusters, key=len)
        
        # Aggregate only the largest cluster
        filtered_updates = {cid: client_updates[cid] for cid in largest_cluster}
        
        # Use simple averaging within the cluster
        param_names = list(next(iter(filtered_updates.values())).keys())
        aggregated_params = {}
        
        for param_name in param_names:
            param_sum = None
            count = 0
            
            for client_id, update in filtered_updates.items():
                if param_name in update:
                    if param_sum is None:
                        param_sum = update[param_name].clone()
                    else:
                        param_sum += update[param_name]
                    count += 1
            
            if count > 0:
                aggregated_params[param_name] = param_sum / count
        
        return aggregated_params


class NoDefenseServer(FederatedServer):
    """
    No defense - simple federated averaging
    Used for baseline vulnerability assessment
    """
    
    def aggregate_updates(self, client_updates: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Simple federated averaging without any defense
        """
        if not client_updates:
            return {}
        
        param_names = list(next(iter(client_updates.values())).keys())
        aggregated_params = {}
        
        for param_name in param_names:
            param_sum = None
            count = 0
            
            for client_id, update in client_updates.items():
                if param_name in update:
                    if param_sum is None:
                        param_sum = update[param_name].clone()
                    else:
                        param_sum += update[param_name]
                    count += 1
            
            if count > 0:
                aggregated_params[param_name] = param_sum / count
        
        return aggregated_params


def create_defense_server(defense_type: str, model, clients, test_dataset, device, config):
    """
    Factory function to create appropriate defense server
    """
    defense_type = defense_type.lower()
    
    if defense_type == 'none' or defense_type == 'no_defense':
        return NoDefenseServer(model, clients, test_dataset, device, config)
    elif defense_type == 'trimmed_mean':
        return TrimmedMeanServer(model, clients, test_dataset, device, config)
    elif defense_type == 'median':
        return MedianServer(model, clients, test_dataset, device, config)
    elif defense_type == 'coordinate_median':
        return CoordinateMedianServer(model, clients, test_dataset, device, config)
    elif defense_type == 'flame':
        return FLAMEServer(model, clients, test_dataset, device, config)
    elif defense_type == 'krum':
        from federated.server import KrumServer
        return KrumServer(model, clients, test_dataset, device, config)
    elif defense_type == 'rbbd':
        from federated.server import RBBDServer
        return RBBDServer(model, clients, test_dataset, device, config)
    else:
        raise ValueError(f"Unknown defense type: {defense_type}") 