import torch
from torch import nn


class SeqNet(nn.Module):
    def __init__(self, n_features, model_type="lstm", hidden_size=32):
        super().__init__()
        if model_type == "lstm":
            self.rnn = nn.LSTM(n_features, hidden_size, batch_first=True)
        else:
            self.rnn = nn.RNN(n_features, hidden_size, nonlinearity="tanh", batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])
