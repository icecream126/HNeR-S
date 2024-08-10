import torch
from torch import nn
from math import ceil

from .model import MODEL, INP_MODEL
from utils.posenc import EQUIREC_ENC


class EQUIREC(nn.Module):
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
        init_a,
        init_b,
        skip,
        base_resol,
        great_circle,
        pole_singularity,
        east_west,
        gauss_scale,
        time_resolution,
        upscale_factor=None,
        time_enc_type=None,
        mapping_size=None,
        n_levels=2,
        n_features_per_level=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_resol = base_resol
        self.init_a = init_a
        self.init_b = init_b
        self.skip = skip
        self.great_circle = great_circle
        self.upscale_factor = upscale_factor
        self.pole_singularity = pole_singularity
        self.east_west = east_west
        self.hidden_layers = hidden_layers
        self.time_enc_type = time_enc_type
        self.scale = gauss_scale
        self.mapping_size = mapping_size
        self.time_resolution = time_resolution
        self.posenc = EQUIREC_ENC(
            time_resolution=self.time_resolution,
            time=self.time,
            init_a=self.init_a,
            init_b=self.init_b,
            base_resol=self.base_resol,
            upscale_factor=self.upscale_factor,
            great_circle=self.great_circle,
            pole_singularity=self.pole_singularity,
            east_west=self.east_west,
            F=n_features_per_level,
            n_levels=n_levels,
        )
        self.posenc_dim = n_levels * n_features_per_level
        if self.time:
            self.posenc_dim += 1

        self.nonlin = EQUIREC

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
        init_a,
        init_b,
        skip,
        base_resol,
        great_circle,
        pole_singularity,
        east_west,
        upscale_factor,
        n_levels=2,
        n_features_per_level=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_resol = base_resol
        self.init_a = init_a
        self.init_b = init_b
        self.skip = skip
        self.great_circle = great_circle
        self.upscale_factor = upscale_factor
        self.pole_singularity = pole_singularity
        self.east_west = east_west
        self.hidden_layers = hidden_layers
        self.posenc = EQUIREC_ENC(
            time=self.time,
            init_a=self.init_a,
            init_b=self.init_b,
            base_resol=self.base_resol,
            upscale_factor=self.upscale_factor,
            great_circle=self.great_circle,
            pole_singularity=self.pole_singularity,
            east_west=self.east_west,
            F=n_features_per_level,
            n_levels=n_levels,
            input_dim=self.input_dim,
        )
        self.posenc_dim = n_levels * n_features_per_level
        if self.time:
            self.posenc_dim += 1

        self.nonlin = EQUIREC

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
