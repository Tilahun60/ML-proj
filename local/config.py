"""
Configuration file for Network Log Anomaly Detection Project - LOCAL VERSION
"""

# Data paths (relative to local/ directory)
DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Dataset configuration
DATASET_NAME = "CICIDS2017"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# File selection: Set to None to use all files, or specify list of filenames
# RECOMMENDED: Use single DDoS file for best balance (6.5GB memory)
SELECTED_FILES = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
# Alternative options:
#   - ["Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"] (8.3GB, PortScan attacks)
#   - ["Wednesday-workingHours.pcap_ISCX.csv"] (20GB, multiple attack types)
#   - None (use all files, requires ~55GB)

# Feature engineering
SEQUENCE_LENGTH = 50  # Time window size for time-series (reduced for memory efficiency)
NORMALIZE_FEATURES = True
HANDLE_IMBALANCE = True  # Use class weights or SMOTE
MAX_SAMPLES = 500000  # Limit dataset size for memory efficiency (None = use all, set to int for sampling)
# Note: With 2.8M samples and sequence_length=50, full dataset needs ~55GB RAM
# Set to a smaller number (e.g., 500000) if you have limited memory

# Transformer model hyperparameters
TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_length": 50  # Should match SEQUENCE_LENGTH
}

# LSTM model hyperparameters
LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": True
}

# Random Forest hyperparameters
RF_CONFIG = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42
}

# Training configuration - LOCAL (CPU/GPU)
TRAINING_CONFIG = {
    "batch_size": 64,  # Increase to 128-256 if you have GPU
    "learning_rate": 0.001,
    "num_epochs": 50,
    "early_stopping_patience": 10,
    "device": "cuda"  # Will fallback to "cpu" if CUDA not available
}

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

