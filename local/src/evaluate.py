"""
Evaluation utilities for all models
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from torch.utils.data import DataLoader
from src.train import TimeSeriesDataset


def evaluate_deep_model(model, X_test, y_test, device, batch_size=64):
    """
    Evaluate deep learning model (Transformer or LSTM)
    
    Args:
        model: Trained PyTorch model
        X_test: Test sequences (n_samples, seq_len, n_features)
        y_test: Test labels (n_samples,)
        device: PyTorch device
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of metrics and predictions
    """
    model.eval()
    model = model.to(device)
    
    # Create dataloader
    test_dataset = TimeSeriesDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            
            # Get predictions
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs)
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels,
        'confusion_matrix': cm
    }


def evaluate_all_models(transformer_model, lstm_model, rf_model, 
                       X_test, y_test, device, batch_size=64):
    """
    Evaluate all three models and return comparison
    
    Args:
        transformer_model: Trained Transformer model
        lstm_model: Trained LSTM model
        rf_model: Trained Random Forest model
        X_test: Test sequences
        y_test: Test labels
        device: PyTorch device
        batch_size: Batch size for deep models
        
    Returns:
        Dictionary with results for each model
    """
    results = {}
    
    # Evaluate Transformer
    print("\nEvaluating Transformer model...")
    results['transformer'] = evaluate_deep_model(
        transformer_model, X_test, y_test, device, batch_size
    )
    
    # Evaluate LSTM
    print("Evaluating LSTM model...")
    results['lstm'] = evaluate_deep_model(
        lstm_model, X_test, y_test, device, batch_size
    )
    
    # Evaluate Random Forest
    print("Evaluating Random Forest model...")
    results['random_forest'] = {
        'metrics': rf_model.evaluate(X_test, y_test),
        'predictions': rf_model.predict(X_test),
        'probabilities': rf_model.predict_proba(X_test)[:, 1],
        'true_labels': y_test,
        'confusion_matrix': confusion_matrix(
            y_test, rf_model.predict(X_test)
        )
    }
    
    return results


def print_comparison_table(results):
    """Print comparison table of all models"""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-"*80)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name.capitalize():<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} "
              f"{metrics['roc_auc']:<12.4f}")
    
    print("="*80)

