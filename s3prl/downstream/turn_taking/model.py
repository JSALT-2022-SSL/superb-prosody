import torch
import torch.nn as nn

'''
reference: https://github.com/aguirrediego/LSTM-General-Turn-Taking-Model/blob/master/improved_model.py
reference: https://www.cs.utep.edu/nigel/papers/lstm-tt.pdf
We change it from tensorflow to pytorch.
'''

class Model(nn.Module):

    def __init__(self, input_dim, output_dim, dropout, hidden_size, **kwargs):
        super(Model, self).__init__()
        # first PReLU layer
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        # second PReLU layer
        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.PReLU(),
            nn.Dropout(dropout)
        )
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            batch_first=True, bidirectional=False)
        # PReLU layer with clipping
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Sigmoid(),
            nn.PReLU()
        )
        
    def forward(self, features):
        # input shape: (batch_size, timesteps, input_dim)
        timesteps = features.shape[1]
        input_dim = features.shape[2]

        # shape: (batch_size * timesteps, input_dim)
        features = torch.reshape(features, (-1, input_dim))
        x = self.fc1(features)
        x = self.fc2(x)
        
        # shape: (batch_size, timesteps, input_dim)
        x = torch.reshape(x, (-1, timesteps, input_dim))
        x, _  = self.lstm(x, None)

        # shape: (batch_size, timesteps, output_dim)
        out = self.fc3(x)
        
        return out
