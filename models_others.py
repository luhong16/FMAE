import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)
        #self.fc = nn.Linear(hidden_dim, output_dim)
        

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x, mask_channel=None):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.fc2(torch.relu(self.fc1(out[:, -1, :])))
        #out = self.fc(out[:, -1, :])
        return out