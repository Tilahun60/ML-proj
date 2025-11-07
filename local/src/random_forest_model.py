"""
Random Forest baseline model for anomaly detection
Note: Random Forest doesn't use sequences, so we flatten the data
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


class RandomForestAnomalyDetector:
    """
    Random Forest-based anomaly detector
    Flattens time-series sequences for traditional ML approach
    """
    
    def __init__(self, n_estimators=100, max_depth=20, 
                 min_samples_split=5, min_samples_leaf=2, random_state=42):
        """
        Initialize Random Forest model
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.is_trained = False
    
    def flatten_sequences(self, X_seq):
        """
        Flatten time-series sequences for Random Forest
        
        Args:
            X_seq: Sequence array (n_samples, seq_len, n_features)
            
        Returns:
            Flattened array (n_samples, seq_len * n_features)
        """
        n_samples, seq_len, n_features = X_seq.shape
        return X_seq.reshape(n_samples, seq_len * n_features)
    
    def train(self, X_train_seq, y_train):
        """
        Train Random Forest model
        
        Args:
            X_train_seq: Training sequences (n_samples, seq_len, n_features)
            y_train: Training labels (n_samples,)
        """
        print("Training Random Forest model...")
        
        # Flatten sequences
        X_train_flat = self.flatten_sequences(X_train_seq)
        
        print(f"Training on {len(X_train_flat)} samples with {X_train_flat.shape[1]} features")
        
        # Train model
        self.model.fit(X_train_flat, y_train)
        self.is_trained = True
        
        print("Random Forest training completed!")
    
    def predict(self, X_seq):
        """
        Predict classes
        
        Args:
            X_seq: Sequence array (n_samples, seq_len, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_flat = self.flatten_sequences(X_seq)
        return self.model.predict(X_flat)
    
    def predict_proba(self, X_seq):
        """
        Predict probabilities
        
        Args:
            X_seq: Sequence array (n_samples, seq_len, n_features)
            
        Returns:
            Probabilities (n_samples, 2)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_flat = self.flatten_sequences(X_seq)
        return self.model.predict_proba(X_flat)
    
    def evaluate(self, X_seq, y_true):
        """
        Evaluate model performance
        
        Args:
            X_seq: Sequence array (n_samples, seq_len, n_features)
            y_true: True labels (n_samples,)
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X_seq)
        y_proba = self.predict_proba(X_seq)[:, 1]  # Probability of anomaly
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

