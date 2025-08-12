"""
Federated Learning Server Implementations
Base server class and specialized implementations for RBBD and Krum defenses
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Optional, Tuple
import numpy as np
import random
import copy
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defense.tail_region_analyzer import TailRegionAnalyzer
from defense.update_analyzer import UpdateAnalyzer
from defense.rbbd_defense import RBBDDefense
from defense.krum_defense import KrumDefense
from utils.metrics import MetricsCalculator, PerformanceTracker


class FederatedServer:
    """
    Base federated learning server
    Handles client coordination and model aggregation
    """
    
    def __init__(self, global_model: nn.Module, clients: List, 
                 test_dataset: torch.utils.data.Dataset, device: torch.device, config):
        """
        Initialize federated server
        
        Args:
            global_model: Global model to train
            clients: List of client instances
            test_dataset: Test dataset for evaluation
            device: Device to run computations on
            config: Configuration object
        """
        self.global_model = global_model.to(device)
        self.clients = clients
        self.test_dataset = test_dataset
        self.device = device
        self.config = config
        
        # Training parameters
        self.clients_per_round = getattr(config, 'CLIENTS_PER_ROUND', 10)
        self.total_rounds = getattr(config, 'TOTAL_ROUNDS', 100)
        
        # Metrics and tracking
        self.metrics_calculator = MetricsCalculator(device)
        self.performance_tracker = PerformanceTracker()
        
        # Data loaders for evaluation
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        self.backdoor_test_loader = None
        
        # Training history
        self.round_results = []
        
    def set_backdoor_test_dataset(self, backdoor_dataset: torch.utils.data.Dataset):
        """
        Set backdoor test dataset for attack success rate evaluation
        
        Args:
            backdoor_dataset: Dataset containing backdoored test samples
        """
        if backdoor_dataset and len(backdoor_dataset) > 0:
            self.backdoor_test_loader = DataLoader(backdoor_dataset, batch_size=256, shuffle=False)
        else:
            self.backdoor_test_loader = None
    
    def select_clients(self, round_num: int) -> List:
        """
        Select clients for current training round
        
        Args:
            round_num: Current training round number
            
        Returns:
            List of selected clients
        """
        return random.sample(self.clients, k=min(self.clients_per_round, len(self.clients)))
    
    def aggregate_updates(self, client_deltas: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates (simple averaging)
        Override in subclasses for different aggregation strategies
        
        Args:
            client_deltas: Dictionary mapping client IDs to parameter deltas
            
        Returns:
            Aggregated parameter update
        """
        if not client_deltas:
            return {}
        
        # Simple averaging
        aggregated_delta = {}
        num_clients = len(client_deltas)
        
        # Initialize aggregated delta
        sample_delta = next(iter(client_deltas.values()))
        for param_name in sample_delta.keys():
            aggregated_delta[param_name] = torch.zeros_like(sample_delta[param_name])
        
        # Sum all deltas
        for delta in client_deltas.values():
            for param_name, param_delta in delta.items():
                aggregated_delta[param_name] += param_delta
        
        # Average
        for param_name in aggregated_delta.keys():
            aggregated_delta[param_name] /= num_clients
        
        return aggregated_delta
    
    def apply_update(self, aggregated_delta: Dict[str, torch.Tensor]):
        """
        Apply aggregated update to global model
        
        Args:
            aggregated_delta: Aggregated parameter update
        """
        with torch.no_grad():
            for param_name, param in self.global_model.named_parameters():
                if param_name in aggregated_delta:
                    param.data += aggregated_delta[param_name].to(self.device)
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate global model on test datasets
        
        Returns:
            Dictionary containing evaluation metrics
        """
        results = {}
        
        # Clean accuracy
        clean_acc = self.metrics_calculator.calculate_accuracy(self.global_model, self.test_loader)
        results['clean_acc'] = clean_acc
        
        # Attack success rate (if backdoor test set available)
        if self.backdoor_test_loader:
            asr = self.metrics_calculator.calculate_attack_success_rate(
                self.global_model, self.backdoor_test_loader)
            results['asr'] = asr
        
        return results
    
    def run_round(self, round_num: int) -> Dict[str, float]:
        """
        Run a single training round
        
        Args:
            round_num: Current round number
            
        Returns:
            Dictionary containing round results
        """
        # Select clients
        selected_clients = self.select_clients(round_num)
        
        # Get global weights
        global_weights = self.global_model.state_dict()
        
        # Collect client updates
        client_deltas = {}
        for client in selected_clients:
            delta = client.local_train(global_weights)
            client_deltas[client.client_id] = delta
        
        # Aggregate updates
        aggregated_delta = self.aggregate_updates(client_deltas)
        
        # Apply update to global model
        if aggregated_delta:
            self.apply_update(aggregated_delta)
        
        # Evaluate model
        round_results = self.evaluate_global_model()
        round_results['round'] = round_num
        round_results['num_clients'] = len(selected_clients)
        
        # Store results
        self.round_results.append(round_results)
        self.performance_tracker.add_metrics(round_num, round_results)
        
        return round_results
    
    def train(self) -> Dict[str, List]:
        """
        Run complete federated training
        
        Returns:
            Training history dictionary
        """
        print(f"Starting federated training for {self.total_rounds} rounds...")
        
        for round_num in range(1, self.total_rounds + 1):
            round_results = self.run_round(round_num)
            
            # Log progress
            if round_num % getattr(self.config, 'LOG_INTERVAL', 5) == 0 or round_num == self.total_rounds:
                clean_acc = round_results.get('clean_acc', 0.0)
                asr = round_results.get('asr', 0.0)
                print(f"Round {round_num:3d} | Clean Acc: {clean_acc:.2f}% | ASR: {asr:.2f}%")
        
        return self.get_training_history()
    
    def get_training_history(self) -> Dict[str, List]:
        """
        Get training history in format suitable for analysis
        
        Returns:
            Dictionary with lists of metrics over time
        """
        if not self.round_results:
            return {}
        
        history = {}
        keys = self.round_results[0].keys()
        
        for key in keys:
            history[key] = [result[key] for result in self.round_results]
        
        return history


class RBBDServer(FederatedServer):
    """
    Federated server implementing RBBD (Representation-Based Backdoor Defense)
    """
    
    def __init__(self, global_model: nn.Module, clients: List,
                 test_dataset: torch.utils.data.Dataset, device: torch.device, config):
        """
        Initialize RBBD server
        
        Args:
            global_model: Global model to train
            clients: List of client instances
            test_dataset: Test dataset for evaluation
            device: Device to run computations on
            config: Configuration object
        """
        super().__init__(global_model, clients, test_dataset, device, config)
        
        # Initialize RBBD components
        self.tail_analyzer = TailRegionAnalyzer(global_model, device, config)
        self.update_analyzer = UpdateAnalyzer(global_model, self.tail_analyzer, device, config)
        self.rbbd_defense = RBBDDefense(config)
        
        # Analysis data loaders
        self.tail_loader = None
        self.core_loader = None
        self.validation_dataset = None
        
    def setup_analysis_loaders(self, validation_dataset: torch.utils.data.Dataset):
        """
        Setup data loaders for tail/core analysis
        
        Args:
            validation_dataset: Validation dataset for analysis
        """
        self.validation_dataset = validation_dataset
        
        # Create validation loader
        val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
        
        # Identify tail regions
        tail_indices = self.tail_analyzer.identify_tail_regions(val_loader)
        
        if len(tail_indices) == 0:
            print("Warning: No tail samples found. Using random split.")
            # Fallback: create artificial tail/core split
            all_indices = list(range(len(validation_dataset)))
            tail_size = max(1, len(all_indices) // 10)  # 10% as tail
            tail_indices = random.sample(all_indices, tail_size)
        
        # Create tail and core datasets
        all_indices = set(range(len(validation_dataset)))
        core_indices = list(all_indices - set(tail_indices))
        
        if len(core_indices) == 0:
            core_indices = list(range(min(100, len(validation_dataset))))
        
        tail_dataset = Subset(validation_dataset, tail_indices)
        core_dataset = Subset(validation_dataset, core_indices)
        
        self.tail_loader = DataLoader(tail_dataset, batch_size=128, shuffle=False)
        self.core_loader = DataLoader(core_dataset, batch_size=128, shuffle=False)
        
        print(f"Analysis setup: {len(tail_indices)} tail samples, {len(core_indices)} core samples")
        
        # Establish baseline features
        self.tail_analyzer.establish_baseline(self.core_loader)
    
    def run_round(self, round_num: int) -> Dict[str, float]:
        """
        Run training round with RBBD defense
        
        Args:
            round_num: Current round number
            
        Returns:
            Round results including defense statistics
        """
        # Select clients
        selected_clients = self.select_clients(round_num)
        
        # Get global weights
        global_weights = self.global_model.state_dict()
        
        # Collect client updates
        client_deltas = {}
        for client in selected_clients:
            delta = client.local_train(global_weights)
            client_deltas[client.client_id] = delta
        
        # Analyze client updates if analysis loaders are available
        client_analyses = {}
        if self.tail_loader and self.core_loader:
            for client in selected_clients:
                # Create temporary model with client update
                temp_model = copy.deepcopy(self.global_model)
                with torch.no_grad():
                    for param_name, param in temp_model.named_parameters():
                        if param_name in client_deltas[client.client_id]:
                            param.data += client_deltas[client.client_id][param_name].to(self.device)
                
                # Analyze the update
                analysis = self.update_analyzer.analyze_client_update(
                    temp_model, self.tail_loader, self.core_loader, self.core_loader)
                client_analyses[client.client_id] = analysis
        
        # Get current model performance for defense intensity adjustment
        current_metrics = self.evaluate_global_model()
        current_asr = current_metrics.get('asr', 0.0)
        
        # Update defense system
        self.rbbd_defense.update_defense_intensity(current_asr)
        client_categories = self.rbbd_defense.analyze_client_suspicion(client_analyses, round_num)
        
        # Calculate client weights
        client_ids = [c.client_id for c in selected_clients]
        client_weights = self.rbbd_defense.calculate_client_weights(
            client_ids, client_categories, round_num)
        
        # Aggregate updates with RBBD defense
        aggregated_delta = self.rbbd_defense.aggregate_updates(client_deltas, client_weights)
        
        # Apply update to global model
        if aggregated_delta:
            self.apply_update(aggregated_delta)
        
        # Evaluate model
        round_results = current_metrics
        round_results['round'] = round_num
        round_results['num_clients'] = len(selected_clients)
        
        # Add defense statistics
        defense_stats = self.rbbd_defense.get_defense_statistics()
        round_results.update({
            'defense_intensity': defense_stats['defense_intensity'],
            'quarantined_clients': defense_stats['quarantined_clients'],
            'avg_suspicion': defense_stats['avg_suspicion']
        })
        
        # Store results
        self.round_results.append(round_results)
        self.performance_tracker.add_metrics(round_num, round_results)
        
        return round_results


class KrumServer(FederatedServer):
    """
    Federated server implementing Krum defense
    """
    
    def __init__(self, global_model: nn.Module, clients: List,
                 test_dataset: torch.utils.data.Dataset, device: torch.device, config):
        """
        Initialize Krum server
        
        Args:
            global_model: Global model to train
            clients: List of client instances
            test_dataset: Test dataset for evaluation
            device: Device to run computations on
            config: Configuration object
        """
        super().__init__(global_model, clients, test_dataset, device, config)
        
        # Initialize Krum defense
        self.krum_defense = KrumDefense(config)
    
    def aggregate_updates(self, client_deltas: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates using Krum algorithm
        
        Args:
            client_deltas: Dictionary mapping client IDs to parameter deltas
            
        Returns:
            Selected best parameter update
        """
        return self.krum_defense.aggregate_updates(client_deltas)
    
    def run_round(self, round_num: int) -> Dict[str, float]:
        """
        Run training round with Krum defense
        
        Args:
            round_num: Current round number
            
        Returns:
            Round results including Krum statistics
        """
        # Run base round logic
        round_results = super().run_round(round_num)
        
        # Add Krum-specific statistics
        krum_stats = self.krum_defense.get_defense_statistics()
        round_results.update({
            'krum_score': krum_stats.get('avg_krum_score', 0.0),
            'total_selections': krum_stats.get('total_selections', 0)
        })
        
        return round_results