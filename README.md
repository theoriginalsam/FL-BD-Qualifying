# RBBD: Representation-Based Backdoor Defense in Federated Learning

This repository implements the **Representation-Based Backdoor Defense (RBBD)** framework for protecting federated learning systems against sophisticated edge-case backdoor attacks. The implementation is based on the paper "A Representation-Based Framework for Robust Backdoor Defense in Federated Learning".

## 🎯 Key Features

- **Edge-Case Attack Defense**: Protects against backdoor attacks targeting uncommon but legitimate inputs
- **Semantic Analysis**: Uses representation-based detection instead of parameter-focused approaches  
- **Adaptive Thresholding**: Dynamic defense intensity based on threat patterns and convergence
- **Multi-Layer Analysis**: Feature extraction from multiple neural network layers
- **Risk-Based Classification**: Categorizes clients into trust levels with appropriate aggregation weights

## 🏗️ Architecture

```
rbbd_federated_defense/
├── config/                 # Configuration management
├── data/                   # Data loading and partitioning
├── models/                 # Neural network models (ResNet)
├── attacks/                # Edge-case backdoor attack implementation
├── defense/                # RBBD defense components
├── federated/              # Federated learning framework
├── utils/                  # Utilities and visualization
├── experiments/            # Experiment runners
└── main.py                 # Main execution script
```

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.comtheoriginalsam/rbbd-federated-defense.git
cd rbbd-federated-defense
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run a comparison between RBBD and Krum defenses:
```bash
python main.py --experiment comparison --dataset cifar10 --rounds 100
```

Run RBBD defense only:
```bash
python main.py --experiment rbbd --dataset cifar10 --clients 50 --rounds 100
```

Run cross-dataset evaluation:
```bash
python main.py --experiment cross_dataset --rounds 50
```

### Configuration Options

Key command-line arguments:

- `--experiment`: Type of experiment (`rbbd`, `krum`, `comparison`, `cross_dataset`, `scalability`)
- `--dataset`: Dataset to use (`cifar10`, `cifar100`)
- `--clients`: Number of federated clients (default: 50)
- `--rounds`: Number of training rounds (default: 100)
- `--malicious_ratio`: Fraction of malicious clients (default: 0.10)
- `--clients_per_round`: Clients selected per round (default: 10)

## 🧠 Core Components

### Defense Framework

**TailRegionAnalyzer**: Identifies low-density regions in feature space where edge-case attacks operate

**UpdateAnalyzer**: Analyzes client updates for semantic anomalies and representation shifts

**RBBDDefense**: Main defense system with adaptive thresholding and risk-based client classification

### Attack Implementation

**EdgeCaseIdentifier**: Identifies edge cases based on statistical rarity and model uncertainty

**EdgeCaseAttacker**: Implements sophisticated backdoor attacks targeting edge cases

### Federated Learning

**Client**: Handles local training with support for honest and malicious behaviors

**RBBDServer**: Federated server implementing the complete RBBD defense

**KrumServer**: Baseline server implementing Krum defense for comparison

## 📊 Experimental Results

The framework demonstrates significant improvements over existing defenses:

- **Clean Accuracy**: 81.2% (vs 78.4% for Krum)
- **Attack Success Rate**: 2.5% (vs 4.0% for Krum)
- **60% improvement** in backdoor success rate reduction

Key advantages:
- Breaks the traditional security-utility trade-off
- Achieves superior performance in both dimensions
- Robust against adaptive attackers

## 🔬 Research Implementation

This implementation closely follows the paper methodology:

### Mathematical Formulation

**Representation Deviation Score**:
```
S_dev(c_i) = (1/L) * Σ D_KL(f_l^baseline || f_l^c_i)
```

**Edge-Case Impact Score**:
```
S_edge(c_i) = Σ(x∈X_edge) ||f(x; θ+Δθ_i) - f(x; θ)||_2 / Σ(x∈X_normal) ||f(x; θ+Δθ_i) - f(x; θ)||_2
```

**Combined Risk Assessment**:
```
Risk(c_i) = α·S_dev(c_i) + β·S_edge(c_i) + γ·S_tail(c_i)
```

### Defense Components

1. **Multi-layer feature space analysis** capturing semantic changes
2. **Distribution-aware filtering** for tail region protection  
3. **Adaptive thresholding** evolving with threat patterns
4. **Risk-based client classification** with weighted aggregation

## 📈 Performance Analysis

The framework includes comprehensive evaluation tools:

- Training performance visualization
- Defense mechanism comparison plots
- Client risk distribution analysis  
- Feature space visualization
- Convergence analysis

## 🛠️ Customization

### Adding New Datasets

Extend `DatasetLoader` in `data/data_loader.py`:

```python
def get_dataset_params(dataset_name):
    if dataset_name == 'your_dataset':
        return your_mean, your_std, num_classes
```

### Custom Attack Patterns

Implement new triggers in `EdgeCaseAttacker`:

```python
def create_trigger(self, image):
    # Implement your trigger pattern
    return triggered_image
```

### Defense Tuning

Modify defense parameters in `config/config.py`:

```python
class Config:
    TAIL_BIAS_THRESHOLD = 0.03      # Tail region sensitivity
    REP_SHIFT_THRESHOLD = 1.8       # Representation change detection
    DEFENSE_INTENSITY = 0.4         # Initial defense strength
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision 0.13+  
- scikit-learn 1.0+
- matplotlib 3.5+
- numpy 1.21+

See `requirements.txt` for complete dependencies.

## 🔍 Troubleshooting

**CUDA Memory Issues**: Reduce batch size or client count
```bash
python main.py --clients 20 --clients_per_round 5
```

**Convergence Problems**: Adjust learning rate or defense intensity
```bash
python main.py --defense_intensity 0.2
```

**Dataset Loading**: Ensure sufficient disk space for CIFAR downloads

## 📚 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{rbbd2025,
  title={A Representation-Based Framework for Robust Backdoor Defense in Federated Learning},
  author={Poudel, Samir and Upadhyay, Kritagya},
  journal={Your Journal},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on research by Samir Poudel and Kritagya Upadhyay at Middle Tennessee State University
- Implements techniques for defending against edge-case backdoor attacks
- Builds upon federated learning and Byzantine-robust aggregation research

---

For questions or issues, please open a GitHub issue or contact the authors.
