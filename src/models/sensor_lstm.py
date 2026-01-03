import torch
import torch.nn as nn

class SensorLSTM(nn.Module):
    """
    LSTM for sensor time series prediction
    Virtual sensor: predict next sensor reading from past sequence
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            out: (batch, 1) - predicted sensor value
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # FC layers
        out = self.fc(last_hidden)
        
        return out


class SensorTransformer(nn.Module):
    """
    Transformer for sensor prediction (alternative to LSTM)
    """
    
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        """
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Transformer
        out = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Take last position
        last_out = out[:, -1, :]  # (batch, d_model)
        
        # Predict
        pred = self.fc(last_out)
        
        return pred
