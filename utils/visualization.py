"""
Visualization utilities for RBBD Federated Defense
Plotting functions for performance analysis and debugging
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import torch


def plot_training_performance(history: Dict[str, List], 
                            title: str = "Training Performance",
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (14, 5)):
    """
    Plot training performance over time
    
    Args:
        history: Training history dictionary
        title: Plot title
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    rounds = history.get('round', [])
    
    # Plot accuracy metrics
    axes[0].plot(rounds, history.get('clean_acc', []), 
                label='Clean Accuracy', linewidth=2, color='blue')
    
    if 'tail_acc' in history:
        axes[0].plot(rounds, history['tail_acc'], 
                    label='Tail Accuracy', linewidth=2, alpha=0.7, color='green')
    
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot attack success rate
    if 'asr' in history:
        axes[1].plot(rounds, history['asr'], 
                    color='red', linewidth=2, label='Attack Success Rate')
        axes[1].set_title('Backdoor Attack Success')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('ASR (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No ASR data available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Attack Success Rate (No Data)')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_defense_comparison(rbbd_history: Dict[str, List], 
                          krum_history: Dict[str, List],
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)):
    """
    Compare RBBD and Krum defense performance
    
    Args:
        rbbd_history: RBBD training history
        krum_history: Krum training history  
        save_path: Path to save plot (optional)
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Clean accuracy comparison
    axes[0, 0].plot(rbbd_history.get('round', []), rbbd_history.get('clean_acc', []), 
                   label='RBBD', linewidth=2, color='blue')
    axes[0, 0].plot(krum_history.get('round', []), krum_history.get('clean_acc', []), 
                   label='Krum', linewidth=2, color='orange')
    axes[0, 0].set_title('Clean Accuracy Comparison')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Attack Success Rate comparison
    if 'asr' in rbbd_history and 'asr' in krum_history:
        axes[0, 1].plot(rbbd_history.get('round', []), rbbd_history.get('asr', []), 
                       label='RBBD', linewidth=2, color='blue')
        axes[0, 1].plot(krum_history.get('round', []), krum_history.get('asr', []), 
                       label='Krum', linewidth=2, color='orange')
        axes[0, 1].set_title('Attack Success Rate Comparison')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('ASR (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Defense intensity (RBBD only)
    if 'defense_intensity' in rbbd_history:
        axes[1, 0].plot(rbbd_history.get('round', []), rbbd_history.get('defense_intensity', []), 
                       color='purple', linewidth=2)
        axes[1, 0].set_title('RBBD Defense Intensity')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Defense Intensity')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Quarantined clients (RBBD only)
    if 'quarantined_clients' in rbbd_history:
        axes[1, 1].plot(rbbd_history.get('round', []), rbbd_history.get('quarantined_clients', []), 
                       color='red', linewidth=2)
        axes[1, 1].set_title('Quarantined Clients (RBBD)')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Number of Quarantined Clients')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Defense Mechanism Comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_performance_metrics_table(histories: Dict[str, Dict[str, List]], 
                                 save_path: Optional[str] = None):
    """
    Create a performance comparison table
    
    Args:
        histories: Dictionary mapping method names to their histories
        save_path: Path to save plot (optional)
    """
    metrics_data = []
    
    for method_name, history in histories.items():
        if not history:
            continue
            
        final_clean_acc = history.get('clean_acc', [0])[-1] if history.get('clean_acc') else 0
        final_asr = history.get('asr', [0])[-1] if history.get('asr') else 0
        avg_clean_acc = np.mean(history.get('clean_acc', [0])) if history.get('clean_acc') else 0
        avg_asr = np.mean(history.get('asr', [0])) if history.get('asr') else 0
        
        metrics_data.append({
            'Method': method_name,
            'Final Clean Acc (%)': f"{final_clean_acc:.2f}",
            'Final ASR (%)': f"{final_asr:.2f}",
            'Avg Clean Acc (%)': f"{avg_clean_acc:.2f}",
            'Avg ASR (%)': f"{avg_asr:.2f}"
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Performance Comparison Table', fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_client_risk_distribution(rbbd_defense, round_num: int,
                                save_path: Optional[str] = None):
    """
    Plot client risk distribution from RBBD defense
    
    Args:
        rbbd_defense: RBBD defense instance
        round_num: Current round number
        save_path: Path to save plot (optional)
    """
    risk_summary = rbbd_defense.get_client_risk_summary()
    
    if not risk_summary:
        print("No client risk data available")
        return
    
    # Prepare data
    categories = list(risk_summary.keys())
    counts = list(risk_summary.values())
    colors = ['green', 'yellow', 'orange', 'red', 'purple'][:len(categories)]
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(counts, labels=categories, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Client Risk Distribution (Round {round_num})')
    
    # Bar chart
    bars = ax2.bar(categories, counts, color=colors)
    ax2.set_title(f'Client Counts by Risk Level (Round {round_num})')
    ax2.set_ylabel('Number of Clients')
    ax2.set_xlabel('Risk Category')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_space_visualization(tail_analyzer, 
                                   title: str = "Feature Space Visualization",
                                   save_path: Optional[str] = None):
    """
    Visualize feature space with tail regions highlighted
    
    Args:
        tail_analyzer: TailRegionAnalyzer instance
        title: Plot title
        save_path: Path to save plot (optional)
    """
    embeddings = tail_analyzer.visualize_feature_space()
    
    if embeddings is None:
        print("No embeddings available for visualization")
        return
    
    density_scores = tail_analyzer.density_scores
    tail_indices = tail_analyzer.tail_indices
    
    if density_scores is None:
        print("No density scores available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Density-colored embedding
    scatter1 = ax1.scatter(embeddings[:, 0], embeddings[:, 1], 
                          c=density_scores, cmap='viridis', alpha=0.6, s=20)
    ax1.set_title('Density-Colored Feature Space')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    plt.colorbar(scatter1, ax=ax1, label='Density Score')
    
    # Tail regions highlighted
    normal_mask = np.ones(len(embeddings), dtype=bool)
    if tail_indices is not None and len(tail_indices) > 0:
        # Create mask for tail samples
        tail_mask = np.zeros(len(embeddings), dtype=bool)
        valid_tail_indices = tail_indices[tail_indices < len(embeddings)]
        if len(valid_tail_indices) > 0:
            tail_mask[valid_tail_indices] = True
            normal_mask = ~tail_mask
            
            # Plot normal samples
            ax2.scatter(embeddings[normal_mask, 0], embeddings[normal_mask, 1], 
                       c='blue', alpha=0.6, s=20, label='Normal Samples')
            
            # Plot tail samples
            ax2.scatter(embeddings[tail_mask, 0], embeddings[tail_mask, 1], 
                       c='red', alpha=0.8, s=30, label='Tail Samples')
        else:
            ax2.scatter(embeddings[:, 0], embeddings[:, 1], 
                       c='blue', alpha=0.6, s=20, label='All Samples')
    else:
        ax2.scatter(embeddings[:, 0], embeddings[:, 1], 
                   c='blue', alpha=0.6, s=20, label='All Samples')
    
    ax2.set_title('Tail Regions Highlighted')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence_analysis(histories: Dict[str, Dict[str, List]],
                            metric: str = 'clean_acc',
                            save_path: Optional[str] = None):
    """
    Analyze and plot convergence behavior
    
    Args:
        histories: Dictionary mapping method names to histories
        metric: Metric to analyze convergence for
        save_path: Path to save plot (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Plot metric over time
    for i, (method, history) in enumerate(histories.items()):
        if metric in history:
            rounds = history.get('round', list(range(len(history[metric]))))
            values = history[metric]
            
            ax1.plot(rounds, values, label=method, 
                    color=colors[i % len(colors)], linewidth=2)
    
    ax1.set_title(f'{metric.replace("_", " ").title()} Over Time')
    ax1.set_xlabel('Round')
    ax1.set_ylabel(metric.replace('_', ' ').title())
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot convergence rate (moving average of changes)
    window_size = 5
    for i, (method, history) in enumerate(histories.items()):
        if metric in history and len(history[metric]) > window_size:
            values = np.array(history[metric])
            changes = np.abs(np.diff(values))
            
            # Calculate moving average of changes
            moving_avg_changes = []
            for j in range(window_size, len(changes)):
                avg_change = np.mean(changes[j-window_size:j])
                moving_avg_changes.append(avg_change)
            
            rounds = list(range(window_size + 1, len(values)))
            ax2.plot(rounds, moving_avg_changes, label=method,
                    color=colors[i % len(colors)], linewidth=2)
    
    ax2.set_title(f'Convergence Rate ({metric})')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Moving Avg of Changes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_results_summary(histories: Dict[str, Dict[str, List]], 
                        output_file: str = "results_summary.txt"):
    """
    Save numerical results summary to text file
    
    Args:
        histories: Dictionary mapping method names to histories
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        f.write("RBBD Federated Defense - Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for method_name, history in histories.items():
            f.write(f"Method: {method_name}\n")
            f.write("-" * 30 + "\n")
            
            if 'clean_acc' in history:
                clean_acc = history['clean_acc']
                f.write(f"Clean Accuracy:\n")
                f.write(f"  Final: {clean_acc[-1]:.2f}%\n")
                f.write(f"  Average: {np.mean(clean_acc):.2f}%\n")
                f.write(f"  Std: {np.std(clean_acc):.2f}%\n")
            
            if 'asr' in history:
                asr = history['asr']
                f.write(f"Attack Success Rate:\n")
                f.write(f"  Final: {asr[-1]:.2f}%\n")
                f.write(f"  Average: {np.mean(asr):.2f}%\n")
                f.write(f"  Std: {np.std(asr):.2f}%\n")
            
            f.write("\n")
        
        f.write(f"Summary generated from {len(histories)} methods\n")
    
    print(f"Results summary saved to {output_file}")