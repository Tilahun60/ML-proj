"""
Main execution script for Network Log Anomaly Detection Project - LOCAL VERSION
Runs all three models (Transformer, LSTM, Random Forest) and compares results
"""

import os
import sys
import torch
import numpy as np

# Add local directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_preprocessing import DataPreprocessor
from src.transformer_model import TransformerAnomalyDetector
from src.lstm_model import LSTMAnomalyDetector
from src.random_forest_model import RandomForestAnomalyDetector
from src.train import train_model
from src.evaluate import evaluate_all_models, print_comparison_table
from src.visualize import create_all_visualizations


def main():
    """Main execution function"""
    
    print("="*80)
    print("Network Log Anomaly Detection using Transformer on Time-Series Data")
    print("LOCAL VERSION")
    print("="*80)
    
    # Create directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, config.DATA_DIR)
    model_dir = os.path.join(base_dir, config.MODEL_DIR)
    results_dir = os.path.join(base_dir, config.RESULTS_DIR)
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Check for dataset
    print("\n" + "="*80)
    print("STEP 1: Data Preprocessing")
    print("="*80)
    
    # Check for dataset files or directory
    if not os.path.exists(data_dir):
        print(f"\n⚠️  ERROR: Data directory not found: {data_dir}/")
        print("Please create the data directory and place CSV files there.")
        return
    
    dataset_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not dataset_files:
        print(f"\n⚠️  WARNING: No CSV dataset found in {data_dir}/")
        print("Please download the CICIDS2017 dataset and place CSV files in the data/ directory.")
        print("Dataset source: https://www.unb.ca/cic/datasets/ids-2017.html")
        print("\nFor testing purposes, you can use any network log dataset with:")
        print("- Flow-based features (duration, bytes, packets, etc.)")
        print("- A 'Label' or ' Label' column with 'BENIGN' for normal traffic")
        print("\nExiting...")
        return
    
    # Use the data directory - preprocessor will load selected or all CSV files
    dataset_path = data_dir
    print(f"Found {len(dataset_files)} CSV files in {data_dir}/")
    if config.SELECTED_FILES:
        print(f"Using selected file(s): {config.SELECTED_FILES}")
    else:
        print("All files will be combined for training.")
    
    # Preprocess data
    preprocessor = DataPreprocessor(
        sequence_length=config.SEQUENCE_LENGTH,
        normalize=config.NORMALIZE_FEATURES,
        selected_files=config.SELECTED_FILES
    )
    
    data = preprocessor.preprocess_pipeline(
        dataset_path, 
        create_sequences=True,
        max_samples=config.MAX_SAMPLES
    )
    
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
    
    # Training configuration
    train_config = config.TRAINING_CONFIG.copy()
    train_config['model_dir'] = model_dir
    train_config['device'] = device
    
    # Initialize models
    print("\n" + "="*80)
    print("STEP 2: Model Initialization")
    print("="*80)
    
    transformer_model = TransformerAnomalyDetector(
        n_features=n_features,
        **config.TRANSFORMER_CONFIG
    )
    print(f"Transformer model initialized: {sum(p.numel() for p in transformer_model.parameters()):,} parameters")
    
    lstm_model = LSTMAnomalyDetector(
        n_features=n_features,
        **config.LSTM_CONFIG
    )
    print(f"LSTM model initialized: {sum(p.numel() for p in lstm_model.parameters()):,} parameters")
    
    rf_model = RandomForestAnomalyDetector(**config.RF_CONFIG)
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
        model_name="transformer"
    )
    
    # Train LSTM
    print("\n" + "-"*80)
    lstm_history, lstm_model = train_model(
        lstm_model,
        (X_train, y_train),
        (X_val, y_val),
        train_config,
        class_weights=class_weights,
        model_name="lstm"
    )
    
    # Train Random Forest
    print("\n" + "-"*80)
    rf_model.train(X_train, y_train)
    rf_model.save(os.path.join(model_dir, "random_forest.pkl"))
    
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
        save_dir=results_dir
    )
    
    # Save results to file
    print("\n" + "="*80)
    print("STEP 6: Saving Results")
    print("="*80)
    
    results_file = os.path.join(results_dir, "model_comparison.txt")
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name.upper()} MODEL\n")
            f.write("-"*80 + "\n")
            metrics = result['metrics']
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name.capitalize()}: {metric_value:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to {results_file}")
    
    print("\n" + "="*80)
    print("PROJECT COMPLETE!")
    print("="*80)
    print(f"\nAll results and visualizations saved to: {results_dir}/")
    print(f"Trained models saved to: {model_dir}/")


if __name__ == "__main__":
    main()

