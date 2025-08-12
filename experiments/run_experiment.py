"""
Experiment runner for RBBD Federated Defense
Comprehensive experiment setup and execution
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import copy
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.data_loader import DatasetLoader, BackdoorDatasetCreator
from data.data_partitioner import DataPartitioner
from models.resnet_model import get_resnet_model, fine_tune_resnet
from attacks.edge_case_attack import EdgeCaseIdentifier, EdgeCaseAttacker
from federated.client import Client
from federated.server import RBBDServer, KrumServer, FederatedServer
from federated.baseline_defenses import create_defense_server
from utils.metrics import set_seed
from utils.visualization import (plot_training_performance, plot_defense_comparison,
                                  plot_performance_metrics_table, save_results_summary)


class ExperimentRunner:
    """
    Enhanced experiment runner for comprehensive defense evaluation
    """
    
    def __init__(self, config: Config):
        """
        Initialize experiment runner
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Set random seed for reproducibility
        set_seed(getattr(config, 'RANDOM_SEED', 42))
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.train_datasets = None
        self.test_dataset = None
        self.validation_dataset = None
        self.backdoor_train_dataset = None
        self.backdoor_test_dataset = None
        self.edge_case_identifier = None
        self.edge_case_attacker = None
        self.clients = None
        self.defense_server = None
        
        # Results storage
        self.results_history = {}
    
    def setup_experiment(self) -> Tuple[List, torch.utils.data.DataLoader, nn.Module]:
        """
        Setup complete experiment including datasets, model, attack, and clients
        
        Returns:
            Tuple of (train_loaders, test_loader, model)
        """
        print("Setting up comprehensive experiment...")
        
        # Setup datasets
        train_loaders, test_loader = self.setup_datasets()
        
        # Setup model
        model = self.setup_model()
        
        return train_loaders, test_loader, model
    
    def setup_datasets(self) -> Tuple[List, torch.utils.data.DataLoader]:
        """
        Setup datasets for the experiment
        
        Returns:
            Tuple of (train_loaders, test_loader)
        """
        print("Setting up datasets...")
        
        # Create dataset loader
        dataset_loader = DatasetLoader(self.config.DATASET)
        
        # Load and partition data
        train_dataset, test_dataset = dataset_loader.load_datasets()
        
        # Create data partitioner
        partitioner = DataPartitioner(
            train_dataset,
            num_clients=self.config.N_CLIENTS,
            seed=getattr(self.config, 'RANDOM_SEED', 42)
        )
        
        # Partition data among clients using Dirichlet distribution for non-IID
        alpha = getattr(self.config, 'DIRICHLET_ALPHA', 0.5)
        client_datasets = partitioner.dirichlet_partition(alpha=alpha)
        
        # Create data loaders
        train_loaders = []
        for client_data in client_datasets:
            loader = torch.utils.data.DataLoader(
                client_data,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=2
            )
            train_loaders.append(loader)
        
        # Test data loader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
        # Store datasets
        self.train_datasets = client_datasets
        self.test_dataset = test_dataset
        
        # Create validation dataset (subset of test for analysis)
        validation_size = min(1000, len(test_dataset))
        val_indices = random.sample(range(len(test_dataset)), validation_size)
        self.validation_dataset = torch.utils.data.Subset(test_dataset, val_indices)
        
        print(f"Datasets setup complete:")
        print(f"  Training clients: {len(train_loaders)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Validation samples: {len(self.validation_dataset)}")
        
        return train_loaders, test_loader
    
    def setup_model(self) -> nn.Module:
        """
        Setup the neural network model
        
        Returns:
            Initialized model
        """
        print("Setting up model...")
        
        # Determine number of classes based on dataset
        num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'fashion_mnist': 10,
            'svhn': 10
        }.get(self.config.DATASET.lower(), 10)
        
        # Create model
        model = get_resnet_model(
            model_name=getattr(self.config, 'MODEL_NAME', 'resnet18'),
            num_classes=num_classes,
            pretrained=getattr(self.config, 'USE_PRETRAINED', True)
        )
        
        model = model.to(self.device)
        self.model = model
        
        print(f"Model setup complete: {type(model).__name__}")
        return model
    
    def setup_edge_case_attack(self):
        """
        Setup edge-case backdoor attack
        
        Returns:
            EdgeCaseAttacker instance
        """
        print("Setting up edge-case attack...")
        
        # Create edge case identifier
        self.edge_case_identifier = EdgeCaseIdentifier(
            model=copy.deepcopy(self.model),
            device=self.device,
            edge_threshold=getattr(self.config, 'EDGE_CASE_THRESHOLD', 0.05),
            uncertainty_threshold=getattr(self.config, 'UNCERTAINTY_THRESHOLD', 0.7)
        )
        
        # Identify edge cases using the validation dataset directly
        edge_case_indices, analysis_results = self.edge_case_identifier.identify_edge_cases(
            self.validation_dataset,
            batch_size=self.config.BATCH_SIZE
        )
        
        # Create attacker
        self.edge_case_attacker = EdgeCaseAttacker(
            trigger_size=getattr(self.config, 'TRIGGER_SIZE', 3),
            target_class=getattr(self.config, 'TARGET_CLASS', 0)
        )
        
        # Create backdoor datasets
        backdoor_creator = BackdoorDatasetCreator(
            trigger_size=getattr(self.config, 'TRIGGER_SIZE', 3),
            target_class=getattr(self.config, 'TARGET_CLASS', 0)
        )
        
        self.backdoor_train_dataset, self.backdoor_test_dataset = backdoor_creator.create_backdoor_dataset(
            self.validation_dataset,
            edge_case_indices,
            getattr(self.config, 'BACKDOOR_TRAIN_SAMPLES', 1000),
            getattr(self.config, 'BACKDOOR_TEST_SAMPLES', 2000)
        )
        
        print(f"Edge-case attack setup complete:")
        print(f"  Edge cases identified: {len(edge_case_indices)}")
        print(f"  Backdoor training samples: {len(self.backdoor_train_dataset) if self.backdoor_train_dataset else 0}")
        print(f"  Backdoor test samples: {len(self.backdoor_test_dataset) if self.backdoor_test_dataset else 0}")
        
        return self.edge_case_attacker
    
    def setup_federated_clients(self, train_loaders: List, attack) -> List[Client]:
        """
        Setup federated learning clients
        
        Args:
            train_loaders: List of data loaders for each client
            attack: Attack instance
            
        Returns:
            List of Client instances
        """
        print("Setting up federated clients...")
        
        # Create clients using the stored datasets (not loaders)
        clients = []
        for client_id, dataset in enumerate(self.train_datasets):
            client = Client(
                client_id=client_id,
                dataset=dataset,  # Pass dataset, not loader
                model=copy.deepcopy(self.model),
                device=self.device,
                config=self.config
            )
            clients.append(client)
        
        # Determine malicious clients
        num_malicious = max(1, int(self.config.N_CLIENTS * self.config.MAL_RATIO))
        malicious_ids = random.sample(range(self.config.N_CLIENTS), num_malicious)
        
        # Set malicious clients
        for client_id in malicious_ids:
            if client_id < len(clients):
                clients[client_id].set_malicious(self.backdoor_train_dataset)
        
        self.clients = clients
        
        print(f"Federated clients setup complete:")
        print(f"  Total clients: {len(clients)}")
        print(f"  Malicious clients: {num_malicious}")
        print(f"  Malicious client IDs: {sorted(malicious_ids)}")
        
        return clients
    
    def run_federated_training(self, clients: List[Client], model: nn.Module, 
                             test_loader: torch.utils.data.DataLoader, attack) -> Dict[str, any]:
        """
        Run federated training with specified defense
        
        Args:
            clients: List of federated clients
            model: Global model
            test_loader: Test data loader
            attack: Attack instance
            
        Returns:
            Dictionary with training results
        """
        # Create defense server based on configuration
        defense_type = getattr(self.config, 'DEFENSE_TYPE', 'rbbd')
        
        server = create_defense_server(
            defense_type=defense_type,
            model=copy.deepcopy(model),
            clients=clients,
            test_dataset=test_loader.dataset,
            device=self.device,
            config=self.config
        )
        
        # Special setup for RBBD server
        if hasattr(server, 'setup_analysis_loaders'):
            server.setup_analysis_loaders(self.validation_dataset)
        
        # Set backdoor test dataset
        if hasattr(server, 'set_backdoor_test_dataset') and self.backdoor_test_dataset:
            server.set_backdoor_test_dataset(self.backdoor_test_dataset)
        
        # Run training
        print(f"Starting federated training with {defense_type.upper()} defense...")
        start_time = time.time()
        
        history = server.train()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Extract final metrics
        final_clean_accuracy = history['clean_accuracy'][-1] if history['clean_accuracy'] else 0
        final_attack_success_rate = history['attack_success_rate'][-1] if history['attack_success_rate'] else 0
        
        results = {
            'defense_type': defense_type,
            'final_clean_accuracy': final_clean_accuracy,
            'final_attack_success_rate': final_attack_success_rate,
            'training_time': training_time,
            'history': history,
            'rounds_completed': len(history.get('clean_accuracy', []))
        }
        
        # Add round-by-round data if detailed tracking is enabled
        if getattr(self.config, 'DETAILED_TRACKING', False):
            round_by_round = []
            clean_acc_history = history.get('clean_accuracy', [])
            asr_history = history.get('attack_success_rate', [])
            
            for round_num, (clean_acc, asr) in enumerate(zip(clean_acc_history, asr_history), 1):
                round_by_round.append({
                    'round': round_num,
                    'clean_accuracy': clean_acc,
                    'attack_success_rate': asr
                })
            
            results['round_by_round'] = round_by_round
        
        return results
    
    def run_single_experiment(self, defense_type: str) -> Dict[str, any]:
        """
        Run a single experiment with specified defense
        
        Args:
            defense_type: Type of defense to use
            
        Returns:
            Experiment results
        """
        # Set defense type in config
        original_defense = getattr(self.config, 'DEFENSE_TYPE', 'rbbd')
        self.config.DEFENSE_TYPE = defense_type
        
        try:
            # Setup experiment
            train_loaders, test_loader, model = self.setup_experiment()
            attack = self.setup_edge_case_attack()
            clients = self.setup_federated_clients(train_loaders, attack)
            
            # Run training
            results = self.run_federated_training(clients, model, test_loader, attack)
            
            return results
            
        finally:
            # Restore original defense type
            self.config.DEFENSE_TYPE = original_defense
    
    def run_comparison_experiment(self, defense_methods: List[str]) -> Dict[str, Dict[str, any]]:
        """
        Run comparison experiment across multiple defense methods
        
        Args:
            defense_methods: List of defense methods to compare
            
        Returns:
            Comparison results
        """
        print(f"Running comparison experiment with {len(defense_methods)} defense methods...")
        
        comparison_results = {}
        
        for defense_method in defense_methods:
            print(f"\n{'='*20} Testing {defense_method.upper()} {'='*20}")
            
            # Reset random seed for fair comparison
            set_seed(getattr(self.config, 'RANDOM_SEED', 42))
            
            try:
                results = self.run_single_experiment(defense_method)
                comparison_results[defense_method] = results
                
                print(f"{defense_method.upper()} Results:")
                print(f"  Clean Accuracy: {results['final_clean_accuracy']:.2f}%")
                print(f"  Attack Success Rate: {results['final_attack_success_rate']:.2f}%")
                print(f"  Training Time: {results['training_time']:.2f}s")
                
            except Exception as e:
                print(f"Error running {defense_method}: {str(e)}")
                comparison_results[defense_method] = {'error': str(e)}
        
        return comparison_results


# Legacy function compatibility
def run_rbbd_experiment(config: Config) -> Dict[str, List]:
    """Run RBBD experiment (legacy compatibility)"""
    runner = ExperimentRunner(config)
    results = runner.run_single_experiment('rbbd')
    return results['history']


def run_krum_experiment(config: Config) -> Dict[str, List]:
    """Run Krum experiment (legacy compatibility)"""
    runner = ExperimentRunner(config)
    results = runner.run_single_experiment('krum')
    return results['history']


def run_comparison_experiment(config: Config) -> Tuple[Dict[str, List], Dict[str, List]]:
    """Run comparison experiment (legacy compatibility)"""
    runner = ExperimentRunner(config)
    results = runner.run_comparison_experiment(['rbbd', 'krum'])
    
    rbbd_history = results.get('rbbd', {}).get('history', {})
    krum_history = results.get('krum', {}).get('history', {})
    
    return rbbd_history, krum_history


def run_cross_dataset_evaluation(config: Config, datasets: List[str]) -> Dict[str, Dict[str, List]]:
    """Run cross-dataset evaluation"""
    results = {}
    original_dataset = config.DATASET
    
    for dataset in datasets:
        print(f"Testing on {dataset}...")
        config.DATASET = dataset
        
        runner = ExperimentRunner(config)
        dataset_results = runner.run_single_experiment('rbbd')
        results[dataset] = dataset_results['history']
    
    config.DATASET = original_dataset
    return results


def run_scalability_test(config: Config, client_counts: List[int]) -> Dict[str, Dict[str, List]]:
    """Run scalability test with different client counts"""
    results = {}
    original_clients = config.N_CLIENTS
    
    for client_count in client_counts:
        print(f"Testing with {client_count} clients...")
        config.N_CLIENTS = client_count
        config.CLIENTS_PER_ROUND = min(client_count, config.CLIENTS_PER_ROUND)
        
        runner = ExperimentRunner(config)
        scalability_results = runner.run_single_experiment('rbbd')
        results[f'{client_count}_clients'] = scalability_results['history']
    
    config.N_CLIENTS = original_clients
    return results