import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Model, self).__init__()
        self.net = nn.Linear(input_dim, output_dim)
        
    def forward(self, features):
        # input shape: (batch_size, timesteps, input_dim)
        output = self.net(features)
        
        return output
