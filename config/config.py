"""
Configuration file for RBBD Federated Defense
Contains all hyperparameters and settings for the experiment
"""

class Config:
    """Configuration class containing all hyperparameters and settings"""
    
    # ========== Data & Federated Learning Setup ==========
    N_CLIENTS = 50
    MAL_RATIO = 0.10  # 10% of clients are malicious
    DIRICHLET_ALPHA = 0.4  # Controls data non-IID-ness; lower is more non-IID
    VAL_SET_SIZE = 5000  # Size of the server's validation set for analysis
    CLIENTS_PER_ROUND = 10
    TOTAL_ROUNDS = 100
    
    # ========== Client-Side Training ==========
    CLIENT_EPOCHS = 10
    CLIENT_BATCH_SIZE = 128
    CLIENT_LR = 0.01
    CLIENT_MOMENTUM = 0.9
    GRAD_CLIP_NORM = 1.0
    
    # ========== Backdoor Attack Parameters ==========
    TRIGGER_SIZE = 3
    TARGET_CLASS = 0  # 'airplane' in CIFAR-10
    BACKDOOR_TRAIN_SAMPLES = 1000
    BACKDOOR_TEST_SAMPLES = 2000
    
    # ========== Defense System Parameters ==========
    # Part 1: Tail Region Identification
    TAIL_PERCENTILE = 15  # Top 15% least dense points are considered the 'tail'
    TSNE_PERPLEXITY = 30  # Perplexity for t-SNE visualization
    KNN_NEIGHBORS = 15  # k for k-Nearest Neighbors density calculation
    
    # Part 2: Adaptive Defense Logic
    INITIAL_DEFENSE_INTENSITY = 0.4  # Start with a moderate defense level
    INTENSITY_RAMP_UP_RATE = 0.015  # How quickly intensity increases
    INTENSITY_COOL_DOWN_RATE = 0.01  # How quickly intensity decreases
    
    # Part 3: Suspicion Scoring
    EVIDENCE_DECAY = 0.6  # Lower value makes suspicion 'stickier' (less decay)
    TAIL_BIAS_THRESHOLD = 0.03  # Anomaly threshold for tail vs. core impact difference
    REP_SHIFT_THRESHOLD = 1.8  # Anomaly threshold for representation shift
    CONSECUTIVE_ATTACK_STRIKES = 3  # Number of consecutive bad rounds before quarantine
    QUARANTINE_ROUNDS = 1
    
    # ========== Mathematical Formulation Parameters ==========
    # Risk score weights (alpha, beta, gamma)
    ALPHA = 0.4  # Weight for representation deviation score
    BETA = 0.3   # Weight for edge-case impact score
    GAMMA = 0.3  # Weight for tail-bias score
    
    # ========== Risk-Based Client Classification Thresholds ==========
    TAU_LOW = 0.3     # Threshold for trusted clients
    TAU_MEDIUM = 0.6  # Threshold for monitored clients
    TAU_HIGH = 0.8    # Threshold for blacklisted clients
    
    # ========== Krum Defense Parameters ==========
    KRUM_K = 6  # Number of closest clients to consider in Krum
    
    # ========== Fine-tuning Parameters ==========
    FINETUNE_EPOCHS = 3
    FINETUNE_LR = 0.001
    FINETUNE_BATCH_SIZE = 64
    
    # ========== Edge Case Definition Parameters ==========
    EDGE_CASE_THRESHOLD = 0.05  # 5% threshold for statistical rarity
    UNCERTAINTY_THRESHOLD = 0.7  # Threshold for model uncertainty
    
    # ========== Experimental Settings ==========
    RANDOM_SEED = 42
    MAX_SAMPLES_ANALYSIS = 5000  # Maximum samples for feature analysis
    MAX_BATCHES_ANALYSIS = 5     # Maximum batches for representation analysis
    
    # ========== Dataset Settings ==========
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_STD = (0.2675, 0.2565, 0.2761)
    
    # ========== Logging and Evaluation ==========
    LOG_INTERVAL = 5  # Log results every N rounds
    SAVE_MODEL_INTERVAL = 20  # Save model every N rounds
    
    @classmethod
    def get_dataset_params(cls, dataset_name):
        """Get normalization parameters for specific dataset"""
        if dataset_name.lower() == 'cifar10':
            return cls.CIFAR10_MEAN, cls.CIFAR10_STD, 10
        elif dataset_name.lower() == 'cifar100':
            return cls.CIFAR100_MEAN, cls.CIFAR100_STD, 100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    def __repr__(self):
        """String representation of configuration"""
        attrs = []
        for key, value in self.__class__.__dict__.items():
            if not key.startswith('_') and not callable(value):
                attrs.append(f"{key}={value}")
        return f"Config({', '.join(attrs)})"