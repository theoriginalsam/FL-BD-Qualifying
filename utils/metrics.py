"""
Metrics and evaluation utilities for RBBD Federated Defense
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader


class MetricsCalculator:
    """Utility class for calculating various metrics"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    @torch.no_grad()
    def calculate_accuracy(self, model: torch.nn.Module, dataloader: DataLoader) -> float:
        """
        Calculate accuracy of model on given dataset
        
        Args:
            model: Neural network model
            dataloader: DataLoader for evaluation dataset
            
        Returns:
            Accuracy as percentage (0-100)
        """
        model.eval()
        correct = 0
        total = 0
        
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total if total > 0 else 0.0
    
    @torch.no_grad()
    def calculate_attack_success_rate(self, model: torch.nn.Module, 
                                    backdoor_dataloader: DataLoader) -> float:
        """
        Calculate Attack Success Rate (ASR) for backdoor samples
        
        Args:
            model: Neural network model
            backdoor_dataloader: DataLoader containing backdoored samples
            
        Returns:
            Attack success rate as percentage (0-100)
        """
        return self.calculate_accuracy(model, backdoor_dataloader)
    
    @torch.no_grad()
    def calculate_kl_divergence(self, model1: torch.nn.Module, model2: torch.nn.Module,
                               dataloader: DataLoader) -> float:
        """
        Calculate KL divergence between predictions of two models
        
        Args:
            model1: First model (reference)
            model2: Second model (comparison)
            dataloader: DataLoader for evaluation
            
        Returns:
            Average KL divergence
        """
        model1.eval()
        model2.eval()
        total_kl = 0.0
        total_samples = 0
        
        for data, _ in dataloader:
            data = data.to(self.device)
            
            output1 = F.softmax(model1(data), dim=1)
            output2 = F.softmax(model2(data), dim=1)
            
            kl_div = F.kl_div(output2.log(), output1, reduction='none').sum(dim=1)
            total_kl += kl_div.sum().item()
            total_samples += data.size(0)
        
        return total_kl / total_samples if total_samples > 0 else 0.0
    
    @torch.no_grad()
    def calculate_representation_shift(self, model1: torch.nn.Module, model2: torch.nn.Module,
                                     dataloader: DataLoader, layer_name: str = 'avgpool') -> Tuple[float, np.ndarray]:
        """
        Calculate representation shift between two models
        
        Args:
            model1: Reference model
            model2: Comparison model
            dataloader: DataLoader for evaluation
            layer_name: Name of layer to extract features from
            
        Returns:
            Tuple of (average shift magnitude, mean difference vector)
        """
        def extract_features(model, data):
            """Extract features from specified layer"""
            features = []
            
            def hook_fn(module, input, output):
                features.append(output.detach())
            
            # Register hook for ResNet avgpool layer
            if hasattr(model, 'avgpool'):
                handle = model.avgpool.register_forward_hook(hook_fn)
            else:
                # Fallback for other architectures
                handle = list(model.children())[-2].register_forward_hook(hook_fn)
            
            model.eval()
            with torch.no_grad():
                _ = model(data)
            
            handle.remove()
            return features[0] if features else None
        
        model1.eval()
        model2.eval()
        
        all_diffs = []
        
        for data, _ in dataloader:
            data = data.to(self.device)
            
            feat1 = extract_features(model1, data)
            feat2 = extract_features(model2, data)
            
            if feat1 is not None and feat2 is not None:
                # Flatten features
                feat1_flat = torch.flatten(feat1, 1).cpu().numpy()
                feat2_flat = torch.flatten(feat2, 1).cpu().numpy()
                
                diff = feat2_flat - feat1_flat
                all_diffs.append(diff)
        
        if all_diffs:
            all_diffs = np.vstack(all_diffs)
            shift_magnitude = np.linalg.norm(all_diffs, axis=1).mean()
            mean_diff = all_diffs.mean(axis=0)
            return float(shift_magnitude), mean_diff
        else:
            return 0.0, np.array([])
    
    def calculate_edge_case_protection_index(self, clean_acc: float, asr: float, 
                                           false_positive_rate: float = 0.0) -> float:
        """
        Calculate Edge-Case Protection Index (ECPI)
        Balances accuracy against protection while penalizing false positives
        
        Args:
            clean_acc: Clean accuracy (0-100)
            asr: Attack success rate (0-100)
            false_positive_rate: False positive rate (0-1)
            
        Returns:
            ECPI score
        """
        protection_score = 100 - asr  # Higher is better
        utility_score = clean_acc      # Higher is better
        penalty = false_positive_rate * 100  # Higher is worse
        
        # Weighted combination with penalty
        ecpi = 0.5 * utility_score + 0.4 * protection_score - 0.1 * penalty
        return max(0.0, ecpi)
    
    def calculate_adaptive_resilience_score(self, asr_values: List[float], 
                                          clean_acc_values: List[float]) -> float:
        """
        Calculate Adaptive Resilience Score measuring performance against sophisticated attackers
        
        Args:
            asr_values: List of ASR values across different attack strategies
            clean_acc_values: List of clean accuracy values
            
        Returns:
            Adaptive resilience score
        """
        if not asr_values or not clean_acc_values:
            return 0.0
        
        # Lower ASR variance indicates consistent protection
        asr_consistency = 100 - np.std(asr_values)
        
        # Higher average clean accuracy is better
        avg_clean_acc = np.mean(clean_acc_values)
        
        # Lower maximum ASR is better
        max_asr_resistance = 100 - max(asr_values)
        
        # Weighted combination
        ars = 0.4 * avg_clean_acc + 0.3 * max_asr_resistance + 0.3 * asr_consistency
        return max(0.0, ars)


class PerformanceTracker:
    """Class to track and store performance metrics over time"""
    
    def __init__(self):
        self.metrics_history = {
            'round': [],
            'clean_acc': [],
            'asr': [],
            'tail_acc': [],
            'defense_intensity': [],
            'num_quarantined': [],
            'representation_shift': []
        }
    
    def add_metrics(self, round_num: int, metrics: Dict[str, float]):
        """Add metrics for a specific round"""
        self.metrics_history['round'].append(round_num)
        
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get final performance metrics"""
        if not self.metrics_history['round']:
            return {}
        
        final_metrics = {}
        for key, values in self.metrics_history.items():
            if key != 'round' and values:
                final_metrics[f'final_{key}'] = values[-1]
                final_metrics[f'avg_{key}'] = np.mean(values)
                if len(values) > 1:
                    final_metrics[f'std_{key}'] = np.std(values)
        
        return final_metrics
    
    def get_convergence_round(self, metric: str = 'clean_acc', 
                            threshold: float = 0.01, window: int = 5) -> Optional[int]:
        """
        Determine when a metric converged (stopped changing significantly)
        
        Args:
            metric: Metric name to check convergence for
            threshold: Change threshold to consider converged
            window: Window size to check for stability
            
        Returns:
            Round number when convergence occurred, or None
        """
        if metric not in self.metrics_history or len(self.metrics_history[metric]) < window:
            return None
        
        values = self.metrics_history[metric]
        
        for i in range(window, len(values)):
            window_values = values[i-window:i]
            if np.std(window_values) < threshold:
                return self.metrics_history['round'][i-window]
        
        return None


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_parameter_distance(params1: Dict[str, torch.Tensor], 
                               params2: Dict[str, torch.Tensor]) -> float:
    """
    Calculate L2 distance between two sets of parameters
    
    Args:
        params1: First parameter dictionary
        params2: Second parameter dictionary
        
    Returns:
        L2 distance between parameters
    """
    total_dist = 0.0
    total_params = 0
    
    for key in params1.keys():
        if key in params2:
            diff = params1[key] - params2[key]
            total_dist += torch.norm(diff).item() ** 2
            total_params += diff.numel()
    
    return np.sqrt(total_dist / total_params) if total_params > 0 else 0.0