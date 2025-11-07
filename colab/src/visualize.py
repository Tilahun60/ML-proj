"""
Visualization utilities for training curves, confusion matrices, and comparisons
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os


def plot_training_curves(history, model_name, save_dir="results"):
    """
    Plot training and validation curves
    
    Args:
        history: Training history dictionary
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, save_dir="results"):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_model_comparison(results, save_dir="results"):
    """
    Plot comparison bar chart of all models
    
    Args:
        results: Dictionary with results for each model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Extract metric values
    metric_values = {metric: [results[model]['metrics'][metric] for model in models] 
                     for metric in metrics}
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(models, metric_values[metric], 
                     color=['#1f77b4', '#ff7f0e', '#2ca02c'][:len(models)],
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric.upper().replace('_', '-'), fontsize=11)
        ax.set_title(f'{metric.upper().replace("_", "-")} Comparison', 
                    fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {save_path}")
    plt.close()


def plot_roc_curves(results, save_dir="results"):
    """
    Plot ROC curves for all models
    
    Args:
        results: Dictionary with results for each model
        save_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve, auc
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, (model_name, result) in enumerate(results.items()):
        y_true = result['true_labels']
        y_proba = result['probabilities']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                label=f'{model_name.capitalize()} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "roc_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()


def create_all_visualizations(results, transformer_history=None, lstm_history=None, save_dir="results"):
    """
    Create all visualizations
    
    Args:
        results: Dictionary with evaluation results
        transformer_history: Training history for Transformer
        lstm_history: Training history for LSTM
        save_dir: Directory to save plots
    """
    print("\nGenerating visualizations...")
    
    # Plot training curves
    if transformer_history:
        plot_training_curves(transformer_history, "Transformer", save_dir)
    if lstm_history:
        plot_training_curves(lstm_history, "LSTM", save_dir)
    
    # Plot confusion matrices
    for model_name, result in results.items():
        plot_confusion_matrix(
            result['true_labels'],
            result['predictions'],
            model_name.capitalize(),
            save_dir
        )
    
    # Plot comparisons
    plot_model_comparison(results, save_dir)
    plot_roc_curves(results, save_dir)
    
    print("\nAll visualizations saved!")

