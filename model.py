import torch
import torch.nn as nn


class MOSEIdownstream(nn.Module):
    def __init__(self, input_dim, projection_dim, output_class_num, **kwargs):
        super(MOSEIdownstream, self).__init__()
        self.projection = nn.Linear(input_dim, projection_dim)
        self.linear = nn.Linear(projection_dim, output_class_num)

    def forward(self, features):
        features = self.projection(features)
        predicted = self.linear(features)
        return predicted
