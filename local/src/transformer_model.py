"""
Transformer model for time-series anomaly detection
Based on "Attention is All You Need" architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerAnomalyDetector(nn.Module):
    """
    Transformer-based anomaly detector for time-series network logs
    """
    
    def __init__(self, n_features, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, max_seq_length=100):
        """
        Initialize Transformer model
        
        Args:
            n_features: Number of input features
            d_model: Dimension of model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
        """
        super(TransformerAnomalyDetector, self).__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 2)  # Binary classification
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            
        Returns:
            logits: Classification logits (batch_size, 2)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch_size, d_model)
        
        # Global average pooling over sequence dimension
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, 2)
        
        return logits
    
    def predict_proba(self, x):
        """Get probability predictions"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """Get class predictions"""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

