import copy
import torch
from torch import nn
import healpy as hp
from healpy.rotator import Rotator


LAT_MIN = -90.0
LON_MIN = 0.0

LAT_MAX = 90.0
LON_MAX = 360.0


class HealEncoding(nn.Module):
    def __init__(self, time, n_levels, F, great_circle, init_a=-0.1, init_b=0.1):
        super().__init__()
        self.great_circle = great_circle
        self.n_levels = n_levels
        self.n_side = 2 ** (n_levels - 1)
        self.n_pix = hp.nside2npix(self.n_side)
        self.F = F
        self.init_a = init_a
        self.init_b = init_b
        self.time = time
        self.all_level_npix = self.get_all_level_npix(self.n_levels - 1)
        param_tensor = torch.randn(self.all_level_npix, F)
        self.params = nn.Parameter(param_tensor)

        for i in range(n_levels):
            if i != 0:
                start = self.get_all_level_npix(i - 1)
            else:
                start = 0
            end = self.get_all_level_npix(i)
            nn.init.normal_(self.params[start:end], mean=self.init_a, std=self.init_b)

    def get_all_level_npix(self, n_levels):
        sum_npix = 0
        n_levels += 1
        for i in range(n_levels):
            sum_npix += hp.nside2npix(2**i)
        return sum_npix

    def forward(self, x):
        if self.time:
            t = x[..., 2:]
            x = x[..., :2]
        lat_lon = x[..., :2].detach().cpu().numpy()
        lat, lon = lat_lon[..., 0], lat_lon[..., 1]

        all_level_reps = []
        for i in range(self.n_levels):
            neigh_pix, neigh_weight = hp.get_interp_weights(
                nside=2**i, theta=lon, phi=lat, lonlat=True
            )  
            neigh_pix = neigh_pix.flatten()
            neigh_pix = torch.from_numpy(neigh_pix).flatten().to(x.device)
            neigh_weight = torch.from_numpy(neigh_weight).to(x.device)

            if i != 0:
                start = self.get_all_level_npix(i - 1)
            else:
                start = 0
            end = self.get_all_level_npix(i)
            neigh_reps = torch.gather(
                self.params[start:end], 0, neigh_pix.unsqueeze(-1).expand(-1, self.F)
            )  
            neigh_reps = neigh_reps.reshape(4, x.shape[0], self.F)  
            neigh_weight = neigh_weight.unsqueeze(-1).repeat(
                1, 1, self.F
            )  

            neigh_reps = torch.multiply(neigh_reps, neigh_weight)
            neigh_reps = neigh_reps.sum(dim=0)  
            all_level_reps.append(neigh_reps)
        all_level_reps = torch.stack(all_level_reps, dim=-1) 
        all_level_reps = all_level_reps.reshape(x.shape[0], -1)

        if self.time:
            all_level_reps = torch.cat((all_level_reps, t), dim=-1)
        return all_level_reps.float()


class InpHealEncoding(nn.Module):
    def __init__(self, n_levels, F, great_circle, init_a=-0.1, init_b=0.1):
        super().__init__()
        self.great_circle = great_circle
        self.n_levels = n_levels
        self.n_side = 2 ** (n_levels - 1)
        self.n_pix = hp.nside2npix(self.n_side)
        self.F = F
        self.init_a = float(init_a)
        self.init_b = float(init_b)

        self.all_level_npix = self.get_all_level_npix(self.n_levels - 1)
        param_tensor = torch.randn(self.all_level_npix, F)
        self.params = nn.Parameter(param_tensor)

        for i in range(n_levels):

            if i != 0:
                start = self.get_all_level_npix(i - 1)
            else:
                start = 0
            end = self.get_all_level_npix(i)
            nn.init.normal_(self.params[start:end], mean=self.init_a, std=self.init_b)

    def get_all_level_npix(self, n_levels):
        sum_npix = 0
        n_levels += 1
        for i in range(n_levels):
            sum_npix += hp.nside2npix(2**i)
        return sum_npix

    def forward(self, x):
        lat_lon = x[..., :2]
        lat, lon = lat_lon[..., 0], lat_lon[..., 1]

        all_level_reps = []
        for i in range(self.n_levels):
            neigh_pix, neigh_weight = hp.get_interp_weights(
                nside=2**i,
                theta=lat.detach().cpu().numpy(),
                phi=lon.detach().cpu().numpy(),
            )  

            neigh_pix = torch.from_numpy(neigh_pix).flatten().to(x.device)
            neigh_weight = torch.from_numpy(neigh_weight).to(x.device)

            start = (
                self.get_all_level_npix(i - 1) if i != 0 else 0
            )  
            end = self.get_all_level_npix(i)

            neigh_reps = torch.gather(
                self.params[start:end], 0, neigh_pix.unsqueeze(-1).expand(-1, self.F)
            )
            neigh_reps = neigh_reps.view(4, -1, self.F) 
            neigh_weight = neigh_weight.unsqueeze(-1).expand(
                -1, -1, self.F
            )  

            neigh_reps = torch.multiply(neigh_reps, neigh_weight).sum(
                dim=0
            )  
            all_level_reps.append(neigh_reps)

        all_level_reps = torch.stack(all_level_reps, dim=-1).view(x.shape[0], -1)

        return all_level_reps.float()


class EQUIREC_ENC(nn.Module):
    def __init__(
        self,
        great_circle,
        pole_singularity,
        east_west,
        upscale_factor=0.5,
        n_levels=7,
        F=2,
        base_resol=90,
        time=False,
        init_a=-1e-4,
        init_b=1e-4,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.init_a = init_a
        self.init_b = init_b
        self.time = time
        self.F = F
        self.upscale_factor = upscale_factor
        self.great_circle = great_circle
        self.pole_singularity = pole_singularity
        self.east_west = east_west
        self.base_resol = base_resol
        self.level_resols = [
            torch.floor(torch.tensor(self.base_resol * self.upscale_factor**i)).type(
                torch.int64
            )
            for i in range(n_levels)
        ] 
        self.params = nn.ParameterList()

        for i in range(self.n_levels):
            level_lat_dim = self.level_resols[i] + 1
            level_lon_dim = self.level_resols[i] * 2 + 1
            param = nn.Parameter(
                torch.empty(level_lat_dim, level_lon_dim, self.F)
            ) 
            nn.init.uniform_(param, a=-1e-4, b=1e-4)
            self.params.append(param)

    def forward(self, x):
        if self.time:
            time = x[:, 2]
            x = x[:, :2]

        all_level_query_param = []
        dx_dy = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=x.device)

        for i in range(self.n_levels):

            lat, lon = copy.deepcopy(x[:, 0]), copy.deepcopy(x[:, 1])
            lat += 90

            lat_min, lat_max = 0, 180
            lon_min, lon_max = 0, 360
            lat = (lat - lat_min) / (lat_max - lat_min)
            lon = (lon - lon_min) / (lon_max - lon_min) * 2

            self.level_resols[i] = self.level_resols[i].to(x.device)
            lat *= self.level_resols[i]
            lon *= self.level_resols[i]
            query_coord = torch.cat((lat.unsqueeze(-1), lon.unsqueeze(-1)), dim=-1)

            param_coord = torch.floor(query_coord).type(torch.int64)

            neighbor_param_coord = param_coord.unsqueeze(1) + dx_dy
            neighbor_param_coord = neighbor_param_coord.view(
                x.shape[0], -1, 2
            ) 

            torch.clamp(
                neighbor_param_coord[:, :, 0],
                min=0,
                max=self.params[i].shape[0] - 1,
                out=neighbor_param_coord[:, :, 0],
            )
            torch.clamp(
                neighbor_param_coord[:, :, 1],
                min=0,
                max=self.params[i].shape[1] - 1,
                out=neighbor_param_coord[:, :, 1],
            )

            dist = torch.abs(
                torch.sub(
                    query_coord.unsqueeze(1).expand(-1, 4, -1), neighbor_param_coord
                )
            )
            weight = 1.0 - dist
            weight = torch.prod(weight, dim=-1, keepdim=True)

            neighbor_param_index = neighbor_param_coord

            if self.east_west:
                neighbor_param_index[..., 1] = torch.where(
                    neighbor_param_index[..., 1]
                    == torch.tensor(self.params[i].shape[1] - 1),
                    torch.tensor(0, device=x.device),
                    neighbor_param_index[..., 1],
                )  

            if self.pole_singularity:
                neighbor_param_index[..., 1] = torch.where(
                    neighbor_param_index[..., 0] == torch.tensor(0),
                    torch.tensor(0, device=x.device),
                    neighbor_param_index[..., 1],
                )  
                neighbor_param_index[..., 1] = torch.where(
                    neighbor_param_index[..., 0]
                    == torch.tensor(self.params[i].shape[0] - 1),
                    torch.tensor(0, device=x.device),
                    neighbor_param_index[..., 1],
                )  

            neighbor_param_index = neighbor_param_index.view(-1, 2)
            self.params[i] = self.params[i].to(x.device)
            query_param = torch.multiply(
                weight.flatten().unsqueeze(-1).expand(-1, self.F),
                self.params[i][
                    neighbor_param_index[:, 0].long(),
                    neighbor_param_index[:, 1].long(),
                    :,
                ],
            )
            query_param = query_param.view(x.shape[0], -1, self.F)
            query_param = torch.sum(query_param, dim=1)
            all_level_query_param.append(query_param)

        all_level_query_param = torch.cat(all_level_query_param, dim=1)

        if self.time:
            all_level_query_param = torch.cat(
                (all_level_query_param, time.unsqueeze(-1)), dim=-1
            )
        return all_level_query_param
