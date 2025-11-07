"""
Main execution script for Network Log Anomaly Detection Project - COLAB VERSION
Optimized for Google Colab with GPU support
"""

import os
import sys
import torch
import numpy as np
import time

# Add colab directory to path
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
    """Main execution function for Colab"""
    
    print("="*80)
    print("Network Log Anomaly Detection using Transformer on Time-Series Data")
    print("COLAB VERSION - GPU Optimized")
    print("="*80)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU detected - training will be slower!")
        print("   Enable GPU: Runtime -> Change runtime type -> GPU")
    
    # Create directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Check for dataset
    print("\n" + "="*80)
    print("STEP 1: Data Preprocessing")
    print("="*80)
    
    if not os.path.exists(config.DATA_DIR):
        print(f"\n⚠️  ERROR: Data directory not found: {config.DATA_DIR}/")
        print("Please upload CSV files to the data directory.")
        return
    
    dataset_files = [f for f in os.listdir(config.DATA_DIR) if f.endswith('.csv')]
    
    if not dataset_files:
        print(f"\n⚠️  WARNING: No CSV dataset found in {config.DATA_DIR}/")
        print("Please upload the CICIDS2017 CSV file(s) to the data directory.")
        print("\nExiting...")
        return
    
    dataset_path = config.DATA_DIR
    print(f"Found {len(dataset_files)} CSV files in {config.DATA_DIR}/")
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
    train_config['model_dir'] = config.MODEL_DIR
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
    print("Expected time with GPU: ~1.5-2 hours total")
    print("Expected time without GPU: ~4-6 hours total")
    
    start_time = time.time()
    
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
    rf_model.save(os.path.join(config.MODEL_DIR, "random_forest.pkl"))
    
    training_time = time.time() - start_time
    print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
    
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
    
    # Save results to file
    print("\n" + "="*80)
    print("STEP 6: Saving Results")
    print("="*80)
    
    results_file = os.path.join(config.RESULTS_DIR, "model_comparison.txt")
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON RESULTS - COLAB\n")
        f.write("="*80 + "\n\n")
        f.write(f"Training time: {training_time/60:.1f} minutes\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name.upper()} MODEL\n")
            f.write("-"*80 + "\n")
            metrics = result['metrics']
            for metric_name, metric_value in metrics.items():
                f.write(f"{metric_name.capitalize()}: {metric_value:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to {results_file}")
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("PROJECT COMPLETE!")
    print("="*80)
    print(f"\nTotal execution time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"All results and visualizations saved to: {config.RESULTS_DIR}/")
    print(f"Trained models saved to: {config.MODEL_DIR}/")
    print("\nTo download results, use:")
    print("  from google.colab import files")
    print("  files.download('results/model_comparison.png')")


if __name__ == "__main__":
    main()

