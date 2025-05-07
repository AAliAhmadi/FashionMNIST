import torch
import torch.nn as nn

class FashionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]                 # last time step
        return self.fc(out)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # [B,32,28,28]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # [B,32,14,14]
            nn.Conv2d(32, 64, 3, padding=1), # [B,64,14,14]
            nn.ReLU(),
            nn.MaxPool2d(2),                 # [B,64,7,7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # [B,64*7*7]
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
