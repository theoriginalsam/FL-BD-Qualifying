"""
Krum Defense Implementation
Byzantine-robust aggregation method for comparison with RBBD
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class KrumDefense:
    """
    Krum defense implementation for federated learning
    Selects the client update with minimum distance to its k closest neighbors
    """
    
    def __init__(self, config):
        """
        Initialize Krum defense
        
        Args:
            config: Configuration object with Krum parameters
        """
        self.config = config
        self.k = getattr(config, 'KRUM_K', 6)  # Number of closest neighbors to consider
        
        # Statistics tracking
        self.selection_history = []
        self.distance_statistics = []
    
    def flatten_model_params(self, model_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten model parameters into a single vector
        
        Args:
            model_params: Dictionary of model parameters
            
        Returns:
            Flattened parameter vector
        """
        param_vectors = []
        for param in model_params.values():
            param_vectors.append(param.view(-1))
        return torch.cat(param_vectors)
    
    def unflatten_model_params(self, flat_params: torch.Tensor, 
                             param_shapes: Dict[str, torch.Size]) -> Dict[str, torch.Tensor]:
        """
        Unflatten parameter vector back to model parameter dictionary
        
        Args:
            flat_params: Flattened parameter vector
            param_shapes: Dictionary mapping parameter names to shapes
            
        Returns:
            Dictionary of model parameters
        """
        unflattened = {}
        start_idx = 0
        
        for name, shape in param_shapes.items():
            param_size = np.prod(shape)
            end_idx = start_idx + param_size
            unflattened[name] = flat_params[start_idx:end_idx].view(shape)
            start_idx = end_idx
        
        return unflattened
    
    def calculate_pairwise_distances(self, client_deltas: List[Dict[str, torch.Tensor]]) -> np.ndarray:
        """
        Calculate pairwise L2 distances between client updates
        
        Args:
            client_deltas: List of client parameter updates
            
        Returns:
            Distance matrix
        """
        num_clients = len(client_deltas)
        
        # Flatten all client updates
        flat_deltas = []
        for delta in client_deltas:
            flat_delta = self.flatten_model_params(delta)
            flat_deltas.append(flat_delta)
        
        # Calculate pairwise distances
        distance_matrix = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                # L2 distance between flattened parameter vectors
                dist = torch.norm(flat_deltas[i] - flat_deltas[j], p=2).item()
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric matrix
        
        return distance_matrix
    
    def calculate_krum_scores(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Calculate Krum scores for each client
        
        Args:
            distance_matrix: Pairwise distance matrix
            
        Returns:
            Array of Krum scores (lower is better)
        """
        num_clients = distance_matrix.shape[0]
        scores = np.zeros(num_clients)
        
        # Ensure k doesn't exceed available clients
        k = min(self.k, num_clients - 1)
        
        for i in range(num_clients):
            # Get distances from client i to all other clients
            distances = distance_matrix[i].copy()
            distances[i] = np.inf  # Exclude self-distance
            
            # Sort distances and sum the k smallest
            sorted_distances = np.sort(distances)
            scores[i] = np.sum(sorted_distances[:k])
        
        return scores
    
    def select_best_client(self, client_deltas: List[Dict[str, torch.Tensor]],
                          client_ids: List[int]) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        Select the best client update using Krum algorithm
        
        Args:
            client_deltas: List of client parameter updates
            client_ids: List of client IDs
            
        Returns:
            Tuple of (selected_client_id, selected_update)
        """
        if not client_deltas:
            raise ValueError("No client deltas provided")
        
        if len(client_deltas) == 1:
            # If only one client, return it
            return client_ids[0], client_deltas[0]
        
        # Calculate pairwise distances
        distance_matrix = self.calculate_pairwise_distances(client_deltas)
        
        # Calculate Krum scores
        krum_scores = self.calculate_krum_scores(distance_matrix)
        
        # Select client with minimum score
        best_client_idx = np.argmin(krum_scores)
        selected_client_id = client_ids[best_client_idx]
        selected_update = client_deltas[best_client_idx]
        
        # Store statistics
        self.selection_history.append({
            'selected_client': selected_client_id,
            'krum_score': krum_scores[best_client_idx],
            'num_clients': len(client_deltas)
        })
        
        self.distance_statistics.append({
            'mean_distance': np.mean(distance_matrix[distance_matrix > 0]),
            'std_distance': np.std(distance_matrix[distance_matrix > 0]),
            'max_distance': np.max(distance_matrix),
            'min_distance': np.min(distance_matrix[distance_matrix > 0]) if np.any(distance_matrix > 0) else 0.0
        })
        
        return selected_client_id, selected_update
    
    def aggregate_updates(self, client_deltas: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using Krum selection
        
        Args:
            client_deltas: Dictionary mapping client IDs to parameter deltas
            
        Returns:
            Selected parameter update (not averaged, just the best one)
        """
        if not client_deltas:
            return {}
        
        client_ids = list(client_deltas.keys())
        deltas_list = list(client_deltas.values())
        
        # Select best update using Krum
        _, selected_update = self.select_best_client(deltas_list, client_ids)
        
        return selected_update
    
    def multi_krum_aggregate(self, client_deltas: Dict[int, Dict[str, torch.Tensor]],
                           m: int = 3) -> Dict[str, torch.Tensor]:
        """
        Multi-Krum aggregation: average the m best clients according to Krum scores
        
        Args:
            client_deltas: Dictionary mapping client IDs to parameter deltas
            m: Number of best clients to average
            
        Returns:
            Averaged parameter update from m best clients
        """
        if not client_deltas:
            return {}
        
        client_ids = list(client_deltas.keys())
        deltas_list = list(client_deltas.values())
        
        if len(deltas_list) <= m:
            # If we have fewer or equal clients than m, average all
            return self._average_updates(deltas_list)
        
        # Calculate distances and scores
        distance_matrix = self.calculate_pairwise_distances(deltas_list)
        krum_scores = self.calculate_krum_scores(distance_matrix)
        
        # Select m best clients
        best_indices = np.argsort(krum_scores)[:m]
        best_deltas = [deltas_list[i] for i in best_indices]
        
        # Average the selected updates
        return self._average_updates(best_deltas)
    
    def _average_updates(self, deltas_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Average multiple parameter updates
        
        Args:
            deltas_list: List of parameter dictionaries to average
            
        Returns:
            Averaged parameter update
        """
        if not deltas_list:
            return {}
        
        if len(deltas_list) == 1:
            return deltas_list[0]
        
        # Initialize averaged update
        averaged_delta = {}
        sample_delta = deltas_list[0]
        
        for param_name in sample_delta.keys():
            averaged_delta[param_name] = torch.zeros_like(sample_delta[param_name])
        
        # Sum all updates
        for delta in deltas_list:
            for param_name, param_delta in delta.items():
                averaged_delta[param_name] += param_delta
        
        # Divide by number of updates
        num_updates = len(deltas_list)
        for param_name in averaged_delta.keys():
            averaged_delta[param_name] /= num_updates
        
        return averaged_delta
    
    def get_defense_statistics(self) -> Dict[str, float]:
        """
        Get statistics about Krum defense performance
        
        Returns:
            Dictionary of defense statistics
        """
        if not self.selection_history:
            return {
                'total_selections': 0,
                'avg_krum_score': 0.0,
                'avg_num_clients': 0.0
            }
        
        krum_scores = [s['krum_score'] for s in self.selection_history]
        num_clients = [s['num_clients'] for s in self.selection_history]
        
        stats = {
            'total_selections': len(self.selection_history),
            'avg_krum_score': np.mean(krum_scores),
            'std_krum_score': np.std(krum_scores),
            'min_krum_score': np.min(krum_scores),
            'max_krum_score': np.max(krum_scores),
            'avg_num_clients': np.mean(num_clients)
        }
        
        # Add distance statistics if available
        if self.distance_statistics:
            mean_distances = [d['mean_distance'] for d in self.distance_statistics]
            stats.update({
                'avg_mean_distance': np.mean(mean_distances),
                'avg_std_distance': np.mean([d['std_distance'] for d in self.distance_statistics]),
                'avg_max_distance': np.mean([d['max_distance'] for d in self.distance_statistics])
            })
        
        return stats
    
    def get_client_selection_frequency(self) -> Dict[int, int]:
        """
        Get frequency of client selections
        
        Returns:
            Dictionary mapping client IDs to selection counts
        """
        selection_counts = {}
        
        for selection in self.selection_history:
            client_id = selection['selected_client']
            selection_counts[client_id] = selection_counts.get(client_id, 0) + 1
        
        return selection_counts
    
    def reset_statistics(self):
        """Reset all collected statistics"""
        self.selection_history = []
        self.distance_statistics = []
    
    def analyze_client_diversity(self, client_deltas: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Analyze diversity among client updates
        
        Args:
            client_deltas: Dictionary mapping client IDs to parameter deltas
            
        Returns:
            Dictionary with diversity metrics
        """
        if len(client_deltas) < 2:
            return {'diversity_score': 0.0, 'avg_distance': 0.0, 'max_distance': 0.0}
        
        deltas_list = list(client_deltas.values())
        distance_matrix = self.calculate_pairwise_distances(deltas_list)
        
        # Remove diagonal (self-distances)
        upper_triangle = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        diversity_metrics = {
            'diversity_score': np.mean(upper_triangle),  # Average pairwise distance
            'avg_distance': np.mean(upper_triangle),
            'std_distance': np.std(upper_triangle),
            'max_distance': np.max(upper_triangle),
            'min_distance': np.min(upper_triangle) if len(upper_triangle) > 0 else 0.0
        }
        
        return diversity_metrics