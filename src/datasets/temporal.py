from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data_dict):
        # Get file path
        self.data_dict = data_dict
        print()

    def __len__(self):
        return self.data_dict["inputs"].shape[0]

    def __getitem__(self, idx):
        data_out = dict()
        data_out["inputs"] = self.data_dict["inputs"][idx]
        data_out["target"] = self.data_dict["target"][idx]
        # data_out["time"] = self.data_dict['time'][idx]
        data_out["target_shape"] = self.data_dict["target_shape"]
        data_out["mean_lat_weight"] = self.data_dict["mean_lat_weight"]
        # data_out["lat"] = self.data_dict['lat']
        # data_out["lon"] = self.data_dict['lon']
        return data_out
