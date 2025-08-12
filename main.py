"""
Main execution script for RBBD Federated Defense
Runs experiments comparing RBBD and Krum defenses against edge-case backdoor attacks
"""

import argparse
import sys
import os
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from experiments.run_experiment import (
    ExperimentRunner, 
    run_rbbd_experiment, 
    run_krum_experiment, 
    run_comparison_experiment,
    run_cross_dataset_evaluation,
    run_scalability_test
)
from utils.metrics import set_seed


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RBBD Federated Defense Experiments')
    
    parser.add_argument('--experiment', type=str, default='comparison',
                      choices=['rbbd', 'krum', 'comparison', 'cross_dataset', 'scalability'],
                      help='Type of experiment to run')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                      choices=['cifar10', 'cifar100'],
                      help='Dataset to use for experiments')
    
    parser.add_argument('--clients', type=int, default=50,
                      help='Number of federated clients')
    
    parser.add_argument('--rounds', type=int, default=100,
                      help='Number of training rounds')
    
    parser.add_argument('--malicious_ratio', type=float, default=0.10,
                      help='Ratio of malicious clients (0.0-1.0)')
    
    parser.add_argument('--clients_per_round', type=int, default=10,
                      help='Number of clients selected per round')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    parser.add_argument('--trigger_size', type=int, default=3,
                      help='Size of backdoor trigger pattern')
    
    parser.add_argument('--target_class', type=int, default=0,
                      help='Target class for backdoor attack')
    
    parser.add_argument('--alpha', type=float, default=0.4,
                      help='Dirichlet alpha for non-IID data distribution')
    
    parser.add_argument('--defense_intensity', type=float, default=0.4,
                      help='Initial defense intensity for RBBD')
    
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    parser.add_argument('--save_models', action='store_true',
                      help='Save trained models')
    
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory to save results')
    
    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """
    Create configuration object from command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Configuration object
    """
    config = Config()
    
    # Update config with command line arguments
    config.N_CLIENTS = args.clients
    config.ROUNDS = args.rounds  # Use ROUNDS instead of TOTAL_ROUNDS
    config.MAL_RATIO = args.malicious_ratio
    config.CLIENTS_PER_ROUND = args.clients_per_round
    config.RANDOM_SEED = args.seed
    config.TRIGGER_SIZE = args.trigger_size
    config.TARGET_CLASS = args.target_class
    config.DIRICHLET_ALPHA = args.alpha
    config.INITIAL_DEFENSE_INTENSITY = args.defense_intensity
    config.DATASET = args.dataset  # Add the missing DATASET attribute
    
    # Add missing attributes for compatibility
    config.BATCH_SIZE = getattr(config, 'CLIENT_BATCH_SIZE', 128)
    config.VERBOSE = args.verbose
    
    return config


def print_experiment_header(experiment_type: str, config: Config, dataset: str):
    """Print experiment information header"""
    print("="*80)
    print("RBBD FEDERATED DEFENSE EXPERIMENT")
    print("="*80)
    print(f"Experiment Type: {experiment_type.upper()}")
    print(f"Dataset: {dataset.upper()}")
    print(f"Clients: {config.N_CLIENTS}")
    print(f"Rounds: {config.ROUNDS}")
    print(f"Malicious Ratio: {config.MAL_RATIO:.1%}")
    print(f"Clients per Round: {config.CLIENTS_PER_ROUND}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print("="*80)


def run_rbbd_only(config: Config, dataset: str, args):
    """Run RBBD experiment only"""
    print_experiment_header("RBBD Only", config, dataset)
    
    print("Running RBBD defense experiment...")
    history = run_rbbd_experiment(config, dataset)
    
    if history:
        final_acc = history.get('clean_acc', [0])[-1] if history.get('clean_acc') else 0
        final_asr = history.get('asr', [0])[-1] if history.get('asr') else 0
        
        print(f"\nRBBD Results:")
        print(f"  Final Clean Accuracy: {final_acc:.2f}%")
        print(f"  Final Attack Success Rate: {final_asr:.2f}%")
    
    return history


def run_krum_only(config: Config, dataset: str, args):
    """Run Krum experiment only"""
    print_experiment_header("Krum Only", config, dataset)
    
    print("Running Krum defense experiment...")
    history = run_krum_experiment(config, dataset)
    
    if history:
        final_acc = history.get('clean_acc', [0])[-1] if history.get('clean_acc') else 0
        final_asr = history.get('asr', [0])[-1] if history.get('asr') else 0
        
        print(f"\nKrum Results:")
        print(f"  Final Clean Accuracy: {final_acc:.2f}%")
        print(f"  Final Attack Success Rate: {final_asr:.2f}%")
    
    return history


def run_comparison(config: Config, dataset: str, args) -> None:
    """Run comparison experiment between RBBD and Krum"""
    print_experiment_header("RBBD vs Krum Comparison", config, dataset)
    
    print("Running comprehensive comparison experiment...")
    rbbd_history, krum_history = run_comparison_experiment(config)
    
    # Print summary comparison
    if rbbd_history and krum_history:
        rbbd_acc = rbbd_history.get('clean_acc', [0])[-1] if rbbd_history.get('clean_acc') else 0
        rbbd_asr = rbbd_history.get('asr', [0])[-1] if rbbd_history.get('asr') else 0
        krum_acc = krum_history.get('clean_acc', [0])[-1] if krum_history.get('clean_acc') else 0
        krum_asr = krum_history.get('asr', [0])[-1] if krum_history.get('asr') else 0
        
        print(f"\n{'='*50}")
        print("FINAL COMPARISON RESULTS")
        print(f"{'='*50}")
        print(f"{'Method':<15} {'Clean Acc (%)':<15} {'ASR (%)':<10}")
        print(f"{'-'*40}")
        print(f"{'RBBD':<15} {rbbd_acc:<15.2f} {rbbd_asr:<10.2f}")
        print(f"{'Krum':<15} {krum_acc:<15.2f} {krum_asr:<10.2f}")
        print(f"{'-'*40}")
        print(f"{'Improvement':<15} {rbbd_acc - krum_acc:<15.2f} {krum_asr - rbbd_asr:<10.2f}")
        print(f"{'='*50}")
    
    return rbbd_history, krum_history


def run_cross_dataset(config: Config, args):
    """Run experiments across multiple datasets"""
    print_experiment_header("Cross-Dataset Evaluation", config, "Multiple")
    
    print("Running cross-dataset evaluation...")
    results = run_cross_dataset_evaluation(config)
    
    # Print cross-dataset summary
    print(f"\n{'='*60}")
    print("CROSS-DATASET RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for dataset, (rbbd_hist, krum_hist) in results.items():
        if rbbd_hist and krum_hist:
            rbbd_acc = rbbd_hist.get('clean_acc', [0])[-1]
            rbbd_asr = rbbd_hist.get('asr', [0])[-1]
            krum_acc = krum_hist.get('clean_acc', [0])[-1]
            krum_asr = krum_hist.get('asr', [0])[-1]
            
            print(f"\n{dataset.upper()}:")
            print(f"  RBBD: {rbbd_acc:.2f}% acc, {rbbd_asr:.2f}% ASR")
            print(f"  Krum: {krum_acc:.2f}% acc, {krum_asr:.2f}% ASR")
            print(f"  Improvement: +{rbbd_acc - krum_acc:.2f}% acc, -{rbbd_asr - krum_asr:.2f}% ASR")
    
    return results


def run_scalability(config: Config, args):
    """Run scalability test with different client counts"""
    print_experiment_header("Scalability Test", config, args.dataset)
    
    client_counts = [20, 50, 100] if config.N_CLIENTS >= 100 else [10, 20, config.N_CLIENTS]
    
    print(f"Running scalability test with client counts: {client_counts}")
    results = run_scalability_test(config, client_counts)
    
    # Print scalability summary
    print(f"\n{'='*50}")
    print("SCALABILITY RESULTS")
    print(f"{'='*50}")
    print(f"{'Clients':<10} {'Runtime (s)':<12} {'Final Acc (%)':<15} {'Final ASR (%)':<12}")
    print(f"{'-'*50}")
    
    for n_clients, (history, runtime) in results.items():
        if history:
            final_acc = history.get('clean_acc', [0])[-1]
            final_asr = history.get('asr', [0])[-1]
            print(f"{n_clients:<10} {runtime:<12.2f} {final_acc:<15.2f} {final_asr:<12.2f}")
        else:
            print(f"{n_clients:<10} {'Failed':<12} {'N/A':<15} {'N/A':<12}")
    
    return results


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create configuration
    config = create_config_from_args(args)
    
    try:
        # Run appropriate experiment
        if args.experiment == 'rbbd':
            results = run_rbbd_only(config, args.dataset, args)
            
        elif args.experiment == 'krum':
            results = run_krum_only(config, args.dataset, args)
            
        elif args.experiment == 'comparison':
            results = run_comparison(config, args.dataset, args)
            
        elif args.experiment == 'cross_dataset':
            results = run_cross_dataset(config, args)
            
        elif args.experiment == 'scalability':
            results = run_scalability(config, args)
            
        else:
            raise ValueError(f"Unknown experiment type: {args.experiment}")
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during experiment: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()