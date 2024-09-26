# code_snippets/pytorch/model_architectures.py

import torch.nn as nn

class SimplePyTorchModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[64, 128], dropout_rates=[0.5, 0.5], output_dim=10):
        super(SimplePyTorchModel, self).__init__()
        layers = []
        prev_dim = input_dim
        for units, dropout in zip(hidden_units, dropout_rates):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = units
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=1))  # for classification
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
