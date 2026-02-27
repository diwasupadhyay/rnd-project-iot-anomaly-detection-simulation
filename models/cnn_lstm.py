"""
CNN-LSTM Model Architecture — CICIoT2023
==========================================
Input : (batch, 10 timesteps, 38 features)
Output: (batch, 3)  →  Normal / DDoS / Botnet

Architecture:
  Conv1D(64)  → Conv1D(128) → MaxPool → Dropout
  LSTM(128)   → LSTM(64)    → Dropout
  Dense(64)   → Softmax(3)
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        n_features   : int = 38,
        n_classes    : int = 3,
        cnn_filters  : list = [64, 128],
        kernel_size  : int = 3,
        lstm_units   : list = [128, 64],
        dropout      : float = 0.3,
        fc_units     : int = 64,
    ):
        super().__init__()

        # ── CNN block ─────────────────────────────────────────────────────────
        # Input: (batch, timesteps=10, features=38)
        # Conv1d expects (batch, channels, length) → permute before/after
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, cnn_filters[0], kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_filters[0]),
            nn.ReLU(),
            nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(cnn_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=1),   # keeps length ~same
            nn.Dropout(dropout),
        )

        # ── LSTM block ────────────────────────────────────────────────────────
        # After CNN: (batch, cnn_filters[1], timesteps) → permute → (batch, timesteps, cnn_filters[1])
        self.lstm1 = nn.LSTM(
            input_size=cnn_filters[1],
            hidden_size=lstm_units[0],
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm_units[0],
            hidden_size=lstm_units[1],
            num_layers=1,
            batch_first=True,
        )
        self.lstm_dropout = nn.Dropout(dropout)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units[1], fc_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_units, n_classes),
        )

    def forward(self, x):
        # x: (batch, timesteps, features)

        # CNN expects (batch, features, timesteps)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)

        # LSTM expects (batch, timesteps, channels)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.lstm_dropout(x)

        # Take last timestep output
        x = x[:, -1, :]

        return self.classifier(x)


def build_model(n_features=38, n_classes=3, device="cuda") -> CNNLSTM:
    model = CNNLSTM(n_features=n_features, n_classes=n_classes)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅  Model built on {device}")
    print(f"    Trainable parameters: {total_params:,}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(device=device)
    print(model)

    # Quick shape test
    dummy = torch.randn(32, 10, 38).to(device)
    out   = model(dummy)
    print(f"\n    Input  shape : {dummy.shape}")
    print(f"    Output shape : {out.shape}  ← should be (32, 3)")
