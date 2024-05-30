import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
    
class unit(nn.Module):
    def __init__(self, num_channels, interm_dim):
        super().__init__()
        self.num_channels = num_channels
        self.interm_dim = interm_dim
        self.gap = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(4 * num_channels, interm_dim)

    def forward(self, x):
        x = self.gap(torch.permute(x, (0, 2, 1)))
        x = torch.permute(x, (0, 2, 1))
        x = x.view(x.size(0), -1)
        out = F.relu(self.fc(x))
        return out
    
class LossNet(nn.Module):
    def __init__(self, depth=12, input_dim=768, reduction_factor=16):
        super(LossNet, self).__init__()
        interm_dim = input_dim // reduction_factor
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(unit(num_channels=input_dim, interm_dim=interm_dim))
        self.linear = nn.Linear(interm_dim * depth, 1)

    def forward(self, x):  # x: embedding hidden state + encoder hidden state
        out = []
        for i in range(len(self.layers)):
            out.append(self.layers[i](x[i]))
        out = torch.cat(out, dim=-1)
        pred_loss = self.linear(out)
        return pred_loss
