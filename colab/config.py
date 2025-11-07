"""
Configuration file for Network Log Anomaly Detection Project - COLAB VERSION
Optimized for Google Colab with GPU support
"""

# Data paths (for Colab - adjust if needed)
DATA_DIR = "/content/ML-proj/data"
MODEL_DIR = "/content/ML-proj/models"
RESULTS_DIR = "/content/ML-proj/results"

# Dataset configuration
DATASET_NAME = "CICIDS2017"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1

# File selection: RECOMMENDED - Use single DDoS file for Colab
SELECTED_FILES = ["Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
# Alternative: Use PortScan file or combine 2-3 files for more diversity

# Feature engineering - Colab optimized
SEQUENCE_LENGTH = 50  # Time window size for time-series
NORMALIZE_FEATURES = True
HANDLE_IMBALANCE = True
MAX_SAMPLES = None  # Use full file (225K samples is fine for Colab GPU)

# Transformer model hyperparameters
TRANSFORMER_CONFIG = {
    "d_model": 128,
    "nhead": 8,
    "num_layers": 3,  # Reduced from 4 for faster training
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_length": 50
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

# Training configuration - COLAB OPTIMIZED (GPU)
TRAINING_CONFIG = {
    "batch_size": 128,  # Larger batch for GPU efficiency
    "learning_rate": 0.001,
    "num_epochs": 30,  # Reduced for faster training (early stopping will stop earlier)
    "early_stopping_patience": 5,  # Faster early stopping
    "device": "cuda"
}

# Evaluation metrics
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

