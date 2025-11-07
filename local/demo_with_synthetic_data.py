"""
Demo script that generates synthetic network log data for testing
Use this if you don't have the CICIDS2017 dataset yet
"""

import os
import numpy as np
import pandas as pd
from src.data_preprocessing import DataPreprocessor
from src.transformer_model import TransformerAnomalyDetector
from src.lstm_model import LSTMAnomalyDetector
from src.random_forest_model import RandomForestAnomalyDetector
from src.train import train_model
from src.evaluate import evaluate_all_models, print_comparison_table
from src.visualize import create_all_visualizations
import config
import torch


def generate_synthetic_data(n_samples=10000, n_features=20, anomaly_rate=0.1):
    """
    Generate synthetic network log data for demonstration
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        anomaly_rate: Proportion of anomalies
        
    Returns:
        DataFrame with synthetic data
    """
    print("Generating synthetic network log data...")
    
    np.random.seed(42)
    
    # Generate normal traffic features
    n_normal = int(n_samples * (1 - anomaly_rate))
    n_anomaly = n_samples - n_normal
    
    # Normal traffic: lower values, more consistent
    normal_data = np.random.normal(50, 10, (n_normal, n_features))
    normal_data = np.abs(normal_data)  # Ensure non-negative
    
    # Anomaly traffic: higher values, more variable
    anomaly_data = np.random.normal(150, 50, (n_anomaly, n_features))
    anomaly_data = np.abs(anomaly_data)
    
    # Combine
    X = np.vstack([normal_data, anomaly_data])
    
    # Create labels
    labels = ['BENIGN'] * n_normal + ['ATTACK'] * n_anomaly
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    labels = [labels[i] for i in indices]
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['Label'] = labels
    
    print(f"Generated {n_samples} samples: {n_normal} normal, {n_anomaly} anomalies")
    
    return df


def main():
    """Run demo with synthetic data"""
    
    print("="*80)
    print("DEMO: Network Log Anomaly Detection with Synthetic Data")
    print("="*80)
    print("\nThis demo uses synthetic data for testing.")
    print("For real results, use the CICIDS2017 dataset with main.py\n")
    
    # Create directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_data(n_samples=5000, n_features=15, anomaly_rate=0.15)
    
    # Save to CSV
    demo_data_path = os.path.join(config.DATA_DIR, "synthetic_demo_data.csv")
    synthetic_df.to_csv(demo_data_path, index=False)
    print(f"\nSynthetic data saved to {demo_data_path}")
    
    # Preprocess data
    print("\n" + "="*80)
    print("STEP 1: Data Preprocessing")
    print("="*80)
    
    preprocessor = DataPreprocessor(
        sequence_length=config.SEQUENCE_LENGTH,
        normalize=config.NORMALIZE_FEATURES
    )
    
    data = preprocessor.preprocess_pipeline(demo_data_path, create_sequences=True)
    
    X_train, y_train = data['train']
    X_val, y_val = data['val']
    X_test, y_test = data['test']
    n_features = data['n_features']
    class_weights = data['class_weights']
    
    print(f"\nData preprocessing complete!")
    print(f"Feature dimension: {n_features}")
    print(f"Sequence length: {config.SEQUENCE_LENGTH}")
    
    # Setup device
    device = torch.device(config.TRAINING_CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Training configuration (reduced for demo)
    train_config = config.TRAINING_CONFIG.copy()
    train_config['model_dir'] = config.MODEL_DIR
    train_config['device'] = device
    train_config['num_epochs'] = 10  # Reduced for demo
    train_config['batch_size'] = 32  # Smaller batch for demo
    
    # Initialize models
    print("\n" + "="*80)
    print("STEP 2: Model Initialization")
    print("="*80)
    
    # Smaller models for demo
    transformer_model = TransformerAnomalyDetector(
        n_features=n_features,
        d_model=64,  # Reduced
        nhead=4,     # Reduced
        num_layers=2,  # Reduced
        dim_feedforward=256,  # Reduced
        dropout=0.1
    )
    print(f"Transformer model initialized: {sum(p.numel() for p in transformer_model.parameters()):,} parameters")
    
    lstm_model = LSTMAnomalyDetector(
        n_features=n_features,
        hidden_size=64,  # Reduced
        num_layers=1,   # Reduced
        dropout=0.2,
        bidirectional=True
    )
    print(f"LSTM model initialized: {sum(p.numel() for p in lstm_model.parameters()):,} parameters")
    
    rf_model = RandomForestAnomalyDetector(
        n_estimators=50,  # Reduced
        max_depth=10,
        random_state=42
    )
    print("Random Forest model initialized")
    
    # Train models
    print("\n" + "="*80)
    print("STEP 3: Model Training")
    print("="*80)
    
    # Train Transformer
    print("\n" + "-"*80)
    transformer_history, transformer_model = train_model(
        transformer_model,
        (X_train, y_train),
        (X_val, y_val),
        train_config,
        class_weights=class_weights,
        model_name="transformer_demo"
    )
    
    # Train LSTM
    print("\n" + "-"*80)
    lstm_history, lstm_model = train_model(
        lstm_model,
        (X_train, y_train),
        (X_val, y_val),
        train_config,
        class_weights=class_weights,
        model_name="lstm_demo"
    )
    
    # Train Random Forest
    print("\n" + "-"*80)
    rf_model.train(X_train, y_train)
    rf_model.save(os.path.join(config.MODEL_DIR, "random_forest_demo.pkl"))
    
    # Evaluate models
    print("\n" + "="*80)
    print("STEP 4: Model Evaluation")
    print("="*80)
    
    results = evaluate_all_models(
        transformer_model,
        lstm_model,
        rf_model,
        X_test,
        y_test,
        device,
        batch_size=train_config['batch_size']
    )
    
    # Print comparison table
    print_comparison_table(results)
    
    # Create visualizations
    print("\n" + "="*80)
    print("STEP 5: Visualization")
    print("="*80)
    
    create_all_visualizations(
        results,
        transformer_history=transformer_history,
        lstm_history=lstm_history,
        save_dir=config.RESULTS_DIR
    )
    
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {config.RESULTS_DIR}/")
    print("\nNote: This was a demo with synthetic data.")
    print("For real results, download CICIDS2017 dataset and run: python main.py")


if __name__ == "__main__":
    main()

