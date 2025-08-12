"""
Update Analyzer for RBBD Defense
Analyzes client updates for semantic anomalies and representation shifts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import copy


class UpdateAnalyzer:
    """
    Analyzes client model updates for semantic anomalies in feature representations
    Implements the core representation-based analysis described in the paper
    """
    
    def __init__(self, global_model: nn.Module, tail_analyzer, device: torch.device, config):
        """
        Initialize update analyzer
        
        Args:
            global_model: Global federated learning model
            tail_analyzer: TailRegionAnalyzer instance
            device: Device to run computations on
            config: Configuration object
        """
        self.global_model = global_model
        self.tail_analyzer = tail_analyzer
        self.device = device
        self.config = config
        
        # Analysis settings
        self.max_batches = getattr(config, 'MAX_BATCHES_ANALYSIS', 5)
        self.rep_shift_threshold = getattr(config, 'REP_SHIFT_THRESHOLD', 1.8)
        self.tail_bias_threshold = getattr(config, 'TAIL_BIAS_THRESHOLD', 0.03)
    
    @torch.no_grad()
    def extract_model_features(self, model: nn.Module, dataloader: DataLoader, 
                              max_batches: Optional[int] = None) -> np.ndarray:
        """
        Extract feature representations from a model using validation data
        
        Args:
            model: Model to extract features from
            dataloader: DataLoader for feature extraction
            max_batches: Maximum number of batches to process
            
        Returns:
            Feature matrix
        """
        if max_batches is None:
            max_batches = self.max_batches
            
        model.eval()
        features = []
        
        for batch_idx, (data, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            data = data.to(self.device)
            
            # Extract features using ResNet architecture
            x = model.conv1(data)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            
            # Global average pooling
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            
            features.append(x.cpu().numpy())
        
        if features:
            return np.vstack(features)
        else:
            return np.empty((0, 2048))  # ResNet50 feature dimension
    
    def calculate_representation_shift(self, client_model: nn.Module, 
                                     test_loader: DataLoader) -> Tuple[float, np.ndarray]:
        """
        Calculate representation shift between global and client models
        Implements the representation deviation score from the paper
        
        Args:
            client_model: Client model to compare
            test_loader: DataLoader for comparison data
            
        Returns:
            Tuple of (shift_magnitude, mean_difference_vector)
        """
        # Extract features from both models
        global_features = self.extract_model_features(self.global_model, test_loader)
        client_features = self.extract_model_features(client_model, test_loader)
        
        if global_features.shape[0] == 0 or client_features.shape[0] == 0:
            return 0.0, np.array([])
        
        # Ensure same number of samples
        min_samples = min(global_features.shape[0], client_features.shape[0])
        global_features = global_features[:min_samples]
        client_features = client_features[:min_samples]
        
        # Calculate difference
        feature_diff = client_features - global_features
        
        # Calculate shift magnitude (L2 norm of differences)
        shift_magnitude = np.linalg.norm(feature_diff, axis=1).mean()
        mean_diff = feature_diff.mean(axis=0)
        
        return float(shift_magnitude), mean_diff
    
    @torch.no_grad()
    def calculate_kl_impact_on_loader(self, client_model: nn.Module, 
                                    dataloader: DataLoader) -> float:
        """
        Calculate KL divergence impact of client model on a specific data loader
        
        Args:
            client_model: Client model to analyze
            dataloader: DataLoader to evaluate impact on
            
        Returns:
            Average KL divergence
        """
        self.global_model.eval()
        client_model.eval()
        
        total_kl = 0.0
        total_samples = 0
        
        for data, _ in dataloader:
            data = data.to(self.device)
            
            # Get predictions from both models
            global_output = self.global_model(data)
            client_output = client_model(data)
            
            # Convert to probabilities
            global_probs = F.softmax(global_output, dim=1)
            client_probs = F.softmax(client_output, dim=1)
            
            # Calculate KL divergence: KL(client || global)
            kl_div = F.kl_div(client_probs.log(), global_probs, reduction='none').sum(dim=1)
            
            total_kl += kl_div.sum().item()
            total_samples += data.size(0)
        
        return total_kl / total_samples if total_samples > 0 else 0.0
    
    def analyze_tail_vs_core_impact(self, client_model: nn.Module, 
                                  tail_loader: DataLoader, 
                                  core_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Analyze differential impact on tail (edge-case) vs core (normal) regions
        Implements the edge-case impact score and tail-bias score from the paper
        
        Args:
            client_model: Client model to analyze
            tail_loader: DataLoader for tail region samples
            core_loader: DataLoader for core region samples
            
        Returns:
            Tuple of (tail_impact, core_impact, differential_impact)
        """
        # Calculate impact on tail region
        tail_impact = self.calculate_kl_impact_on_loader(client_model, tail_loader)
        
        # Calculate impact on core region  
        core_impact = self.calculate_kl_impact_on_loader(client_model, core_loader)
        
        # Calculate differential impact (tail-bias score)
        differential_impact = tail_impact - core_impact
        
        return tail_impact, core_impact, differential_impact
    
    def calculate_combined_risk_score(self, rep_shift: float, edge_impact: float, 
                                    tail_bias: float) -> float:
        """
        Calculate combined risk score using weighted combination
        Implements the risk assessment formula from the paper
        
        Args:
            rep_shift: Representation deviation score
            edge_impact: Edge-case impact score  
            tail_bias: Tail-bias score
            
        Returns:
            Combined risk score
        """
        # Use weights from config (alpha=0.4, beta=0.3, gamma=0.3)
        alpha = getattr(self.config, 'ALPHA', 0.4)
        beta = getattr(self.config, 'BETA', 0.3) 
        gamma = getattr(self.config, 'GAMMA', 0.3)
        
        # Normalize scores to similar scales
        normalized_rep_shift = min(rep_shift / self.rep_shift_threshold, 1.0)
        normalized_edge_impact = min(abs(edge_impact) / 10.0, 1.0)  # Scale edge impact
        normalized_tail_bias = min(abs(tail_bias) / self.tail_bias_threshold, 1.0)
        
        risk_score = (alpha * normalized_rep_shift + 
                     beta * normalized_edge_impact + 
                     gamma * normalized_tail_bias)
        
        return risk_score
    
    def analyze_client_update(self, client_model: nn.Module, 
                            tail_loader: DataLoader,
                            core_loader: DataLoader,
                            test_loader: DataLoader) -> Dict[str, float]:
        """
        Comprehensive analysis of a client model update
        
        Args:
            client_model: Client model to analyze
            tail_loader: DataLoader for tail region samples
            core_loader: DataLoader for core region samples  
            test_loader: DataLoader for general testing
            
        Returns:
            Dictionary containing analysis results
        """
        # Calculate representation shift
        rep_shift, mean_diff = self.calculate_representation_shift(client_model, test_loader)
        
        # Calculate tail vs core impact
        tail_impact, core_impact, diff_impact = self.analyze_tail_vs_core_impact(
            client_model, tail_loader, core_loader)
        
        # Calculate combined risk score
        risk_score = self.calculate_combined_risk_score(rep_shift, tail_impact, diff_impact)
        
        # Determine anomaly flags
        rep_anomaly = rep_shift > self.rep_shift_threshold
        tail_anomaly = abs(diff_impact) > self.tail_bias_threshold
        
        analysis_results = {
            'rep_shift': rep_shift,
            'tail_impact': tail_impact,
            'core_impact': core_impact, 
            'diff_impact': diff_impact,
            'risk_score': risk_score,
            'rep_anomaly': rep_anomaly,
            'tail_anomaly': tail_anomaly,
            'is_suspicious': rep_anomaly or tail_anomaly,
            'mean_diff_norm': np.linalg.norm(mean_diff) if len(mean_diff) > 0 else 0.0
        }
        
        return analysis_results
    
    def batch_analyze_updates(self, client_models: List[nn.Module],
                            client_ids: List[int],
                            tail_loader: DataLoader,
                            core_loader: DataLoader, 
                            test_loader: DataLoader) -> Dict[int, Dict[str, float]]:
        """
        Analyze multiple client updates in batch
        
        Args:
            client_models: List of client models
            client_ids: List of client IDs
            tail_loader: DataLoader for tail samples
            core_loader: DataLoader for core samples
            test_loader: DataLoader for testing
            
        Returns:
            Dictionary mapping client IDs to analysis results
        """
        results = {}
        
        for client_model, client_id in zip(client_models, client_ids):
            analysis = self.analyze_client_update(
                client_model, tail_loader, core_loader, test_loader)
            results[client_id] = analysis
            
        return results
    
    def calculate_baseline_statistics(self, honest_models: List[nn.Module],
                                    test_loader: DataLoader) -> Dict[str, float]:
        """
        Calculate baseline statistics from known honest models
        
        Args:
            honest_models: List of honest client models
            test_loader: DataLoader for evaluation
            
        Returns:
            Dictionary of baseline statistics
        """
        rep_shifts = []
        
        for model in honest_models:
            shift, _ = self.calculate_representation_shift(model, test_loader)
            rep_shifts.append(shift)
        
        if rep_shifts:
            baseline_stats = {
                'mean_rep_shift': np.mean(rep_shifts),
                'std_rep_shift': np.std(rep_shifts),
                'max_rep_shift': np.max(rep_shifts),
                'percentile_75': np.percentile(rep_shifts, 75),
                'percentile_90': np.percentile(rep_shifts, 90)
            }
        else:
            baseline_stats = {
                'mean_rep_shift': 0.0,
                'std_rep_shift': 0.0, 
                'max_rep_shift': 0.0,
                'percentile_75': 0.0,
                'percentile_90': 0.0
            }
        
        return baseline_stats
    
    def adaptive_threshold_update(self, recent_analyses: List[Dict[str, float]], 
                                convergence_status: float) -> Tuple[float, float]:
        """
        Update thresholds adaptively based on recent analyses and convergence
        
        Args:
            recent_analyses: List of recent analysis results
            convergence_status: Convergence status (0-1, higher means more converged)
            
        Returns:
            Tuple of (updated_rep_threshold, updated_tail_threshold)
        """
        if not recent_analyses:
            return self.rep_shift_threshold, self.tail_bias_threshold
        
        # Extract recent scores
        recent_rep_shifts = [a['rep_shift'] for a in recent_analyses]
        recent_tail_biases = [abs(a['diff_impact']) for a in recent_analyses]
        
        # Calculate adaptive thresholds
        base_rep_threshold = self.rep_shift_threshold
        base_tail_threshold = self.tail_bias_threshold
        
        # Adjust based on convergence - stricter when more converged
        convergence_factor = 0.5 + 0.5 * convergence_status
        
        # Adjust based on recent distribution
        if recent_rep_shifts:
            rep_percentile = np.percentile(recent_rep_shifts, 90)
            adaptive_rep_threshold = base_rep_threshold * convergence_factor + rep_percentile * (1 - convergence_factor)
        else:
            adaptive_rep_threshold = base_rep_threshold
            
        if recent_tail_biases:
            tail_percentile = np.percentile(recent_tail_biases, 90)
            adaptive_tail_threshold = base_tail_threshold * convergence_factor + tail_percentile * (1 - convergence_factor)
        else:
            adaptive_tail_threshold = base_tail_threshold
        
        return adaptive_rep_threshold, adaptive_tail_threshold