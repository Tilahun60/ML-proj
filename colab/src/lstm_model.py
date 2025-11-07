"""
LSTM baseline model for time-series anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detector for time-series network logs
    """
    
    def __init__(self, n_features, hidden_size=128, num_layers=2,
                 dropout=0.2, bidirectional=True):
        """
        Initialize LSTM model
        
        Args:
            n_features: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMAnomalyDetector, self).__init__()
        
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # Binary classification
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            
        Returns:
            logits: Classification logits (batch_size, 2)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            h_forward = h_n[-2]  # Last forward hidden state
            h_backward = h_n[-1]  # Last backward hidden state
            h_final = torch.cat([h_forward, h_backward], dim=1)
        else:
            h_final = h_n[-1]  # Last hidden state
        
        # Classification
        logits = self.classifier(h_final)  # (batch_size, 2)
        
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

