import torch
from torch import nn
from math import ceil

from .model import MODEL, INP_MODEL
from utils.posenc import HealEncoding , InpHealEncoding


class HealLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return nn.functional.relu(self.linear(input))


class INR(MODEL):
    def __init__(
        self,
        output_dim,
        hidden_dim,
        hidden_layers,
        skip,
        n_levels,
        n_features_per_level,
        great_circle,
        init_a, 
        init_b,
        mapping_size=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.skip = skip
        self.hidden_layers = hidden_layers
        self.posenc = HealEncoding(time=self.time, n_levels=n_levels, F=n_features_per_level, great_circle=great_circle, init_a = init_a, init_b = init_b)
        self.posenc_dim = n_features_per_level * n_levels
        if self.time:
            self.posenc_dim+=1

        self.nonlin = HealLayer

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(self.posenc_dim, hidden_dim))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_dim + self.posenc_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x = self.posenc(x)
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x


class INP_INR(INP_MODEL):
    def __init__(
        self,
        output_dim,
        hidden_dim,
        hidden_layers,
        skip,
        n_levels,
        n_features_per_level,
        great_circle,
        init_a, 
        init_b,
        mapping_size=256,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.skip = skip
        self.hidden_layers = hidden_layers
        self.posenc = InpHealEncoding(n_levels=n_levels, F=n_features_per_level, great_circle=great_circle, init_a=init_a, init_b=init_b)
        self.posenc_dim = n_features_per_level * n_levels
            

        self.nonlin = HealLayer

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(self.posenc_dim, hidden_dim))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_dim + self.posenc_dim, hidden_dim))
            else:
                self.net.append(self.nonlin(hidden_dim, hidden_dim))

        final_linear = nn.Linear(hidden_dim, output_dim)

        self.net.append(final_linear)

    def forward(self, x):
        x = self.posenc(x)
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x
