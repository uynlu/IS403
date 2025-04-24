import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        dropout_prob: float = 0.3,
        n_steps: int = 1,
    ):
        super(LSTM, self).__init__()

        self.n_steps = n_steps

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(
            in_features=hidden_size,
            out_features=self.n_steps
        )

    def forward(self, x):
        x1, _ = self.lstm(x)
        x2 = self.dropout(x1)
        x3 = self.norm(x2)
        x4 = x3[:, -1, :]
        x5 = self.linear(x4)
        return x5
