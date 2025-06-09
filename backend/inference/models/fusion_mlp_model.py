import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.output_dim = num_classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
