import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
import glob
import netCDF4 as nc
from utils.utils import StandardScalerTorch, MinMaxScalerTorch
import numpy as np
from sklearn.model_selection import train_test_split
import healpy as hp
from PIL import Image


def cmb_sr_preprocess(
    seed, input_dim, dataset_dir, downscale_factor, normalize, zscore_normalize=False
):

    # all dataset processing
    parts = dataset_dir.split("/")
    if "cmb" in dataset_dir:
        parts[1] = "cmb_2048"
    elif "dust" in dataset_dir:
        parts[1] = "dust_2048"
    filename = "/".join(parts) + "/inp_data.npy"

    target = torch.from_numpy(np.load(filename)).unsqueeze(-1).float().to("cuda")

    if normalize:
        scaler = MinMaxScalerTorch()
    elif zscore_normalize:
        scaler = StandardScalerTorch()
    else:
        raise Exception(
            "Set normalize argument properly (normalize or zscore_normalize"
        )

    scaler.fit(target)
    target = scaler.transform(target)

    npix = target.shape[0]

    lat, lon = hp.pix2ang(
        hp.npix2nside(target.shape[0]), torch.arange(npix)
    ) 
    lat, lon = lat.to(torch.float32), lon.to(torch.float32)
    inputs = torch.column_stack((lat, lon))

    mean_lat_weight = torch.abs(torch.cos(lat)).mean()

    target_shape = target.shape
    target = target.reshape(-1, 1)

    inputs = torch.stack([lat, lon], dim=-1)

    all_dict = dict()

    all_dict["inputs"] = inputs
    all_dict["target"] = target
    all_dict["target_shape"] = target_shape
    all_dict["mean_lat_weight"] = mean_lat_weight

    # train dataset processing
    parts = dataset_dir.split("/")
    if "cmb" in dataset_dir:
        if downscale_factor == 2: 
            parts[1] = "cmb_1024"
            filename = "/".join(parts) + "/inp_data.npy"
        elif downscale_factor == 4:  
            parts[1] = "cmb_512"
            filename = "/".join(parts) + "/inp_data.npy"
    elif "dust" in dataset_dir:
        if downscale_factor == 2:  
            parts[1] = "dust_1024"
            filename = "/".join(parts) + "/inp_data.npy"
        elif downscale_factor == 4:  
            parts[1] = "dust_512"
            filename = "/".join(parts) + "/inp_data.npy"

    target = torch.from_numpy(np.load(filename)).float()
    npix = target.shape[0]

    lat, lon = hp.pix2ang(hp.npix2nside(target.shape[0]), torch.arange(npix))
    lat, lon = lat.to(torch.float32), lon.to(torch.float32)
    inputs = torch.column_stack((lat, lon))

    mean_lat_weight = torch.abs(torch.cos(lat)).mean()

    target_shape = target.shape
    target = target.reshape(-1, 1)

    target = scaler.transform(target.to("cuda"))

    inputs = torch.stack([lat, lon], dim=-1)

    X_train, X_val, y_train, y_val = train_test_split(
        inputs, target, test_size=0.2, random_state=seed
    )

    train_dict = dict()
    val_dict = dict()

    train_dict["inputs"] = X_train.to("cuda")
    train_dict["target"] = y_train.to("cuda")
    train_dict["target_shape"] = target_shape
    train_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")

    val_dict["inputs"] = X_val.to("cuda")
    val_dict["target"] = y_val.to("cuda")
    val_dict["target_shape"] = target_shape
    val_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")

    return scaler, train_dict, val_dict, all_dict


def weather_sr_preprocess(
    seed, input_dim, dataset_dir, downscale_factor, normalize, zscore_normalize=False
):
    # all dataset processing
    filename = sorted(glob.glob(os.path.join(dataset_dir, "*")))[0]
    with nc.Dataset(filename, "r") as f:
        for variable in f.variables:
            if variable == "latitude":
                lat = f.variables[variable][:]
            elif variable == "longitude":
                lon = f.variables[variable][:]
            else:
                target = f.variables[variable][0]

    # transform to torch tensor
    target = torch.from_numpy(target)
    lat = torch.from_numpy(lat)
    lon = torch.from_numpy(lon)

    # handle input dimension
    if input_dim == 3:
        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)

    # calculate mean latitude weight
    mean_lat_weight = torch.abs(torch.cos(lat)).mean()
    target_shape = target.shape
    lat, lon = torch.meshgrid(lat, lon)
    lat = lat.flatten()
    lon = lon.flatten()
    target = target.reshape(-1, 1)

    # initialize scaler
    if normalize:
        scaler = MinMaxScalerTorch()
    elif zscore_normalize:
        scaler = StandardScalerTorch()
    else:
        raise Exception(
            "Set normalize argument properly (normalize or zscore_normalize"
        )

    # fit and transform target
    scaler.fit(target.to("cuda"))
    target = scaler.transform(target.to("cuda"))

    inputs = torch.stack([lat, lon], dim=-1)

    all_dict = dict()

    all_dict["inputs"] = inputs.to("cuda")
    all_dict["target"] = target.to("cuda")
    all_dict["target_shape"] = target_shape
    all_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")
    all_dict["lat"] = lat.to("cuda")
    all_dict["lon"] = lon.to("cuda")

    # train dataset processing
    parts = dataset_dir.split("/")
    if downscale_factor == 2: 
        parts[1] = "spatial_0_50"
        filename = "/".join(parts) + "/data.nc"
    elif downscale_factor == 4:  
        parts[1] = "spatial_1_00"
        filename = "/".join(parts) + "/data.nc"

    with nc.Dataset(filename, "r") as f:
        for variable in f.variables:
            if variable == "latitude":
                lat = f.variables[variable][:]
            elif variable == "longitude":
                lon = f.variables[variable][:]
            else:
                target = f.variables[variable][0]

    target = torch.from_numpy(target)
    lat = torch.from_numpy(lat)
    lon = torch.from_numpy(lon)

    if input_dim == 3:
        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)

    mean_lat_weight = torch.abs(torch.cos(lat)).mean() 
    target_shape = target.shape
    lat, lon = torch.meshgrid(lat, lon)
    lat = lat.flatten()
    lon = lon.flatten()

    target = scaler.transform(target.to("cuda"))
    target = target.reshape(-1, 1)

    inputs = torch.stack([lat, lon], dim=-1)

    X_train, X_val, y_train, y_val = train_test_split(
        inputs, target, test_size=0.2, random_state=seed
    )

    train_dict = dict()
    val_dict = dict()

    train_dict["inputs"] = X_train.to("cuda")
    train_dict["target"] = y_train.to("cuda")
    train_dict["target_shape"] = y_train.shape
    train_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")
    train_dict["lat"] = lat.to("cuda")
    train_dict["lon"] = lon.to("cuda")

    val_dict["inputs"] = X_val
    val_dict["target"] = y_val
    val_dict["target_shape"] = y_val.shape
    val_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")
    val_dict["lat"] = lat.to("cuda")
    val_dict["lon"] = lon.to("cuda")
    val_dict["scaler"] = scaler
    return scaler, train_dict, val_dict, all_dict


def weather_regression_preprocess(
    input_dim, test_size, seed, dataset_dir, normalize, zscore_normalize=False, **kwargs
):

    filename = sorted(glob.glob(os.path.join(dataset_dir, "*")))[0]
    with nc.Dataset(filename, "r") as f:
        for variable in f.variables:
            if variable == "latitude":
                lat = f.variables[variable][:]
            elif variable == "longitude":
                lon = f.variables[variable][:]
            else:
                target = f.variables[variable][0]

    target = torch.from_numpy(target)
    lat = torch.from_numpy(lat)
    lon = torch.from_numpy(lon)

    if input_dim == 3:
        lat = torch.deg2rad(lat)
        lon = torch.deg2rad(lon)
    mean_lat_weight = torch.abs(torch.cos(lat)).mean()
    target_shape = target.shape
    lat, lon = torch.meshgrid(lat, lon)
    lat = lat.flatten()
    lon = lon.flatten()
    target = target.reshape(-1, 1)

    if normalize:
        scaler = MinMaxScalerTorch()
    elif zscore_normalize:
        scaler = StandardScalerTorch()
    else:
        raise Exception(
            "Set normalize argument properly (normalize or zscore_normalize"
        )

    target = target.to("cuda")
    scaler.fit(target)

    target = scaler.transform(target)

    input = torch.stack([lat, lon], dim=-1)

    all_dict = dict()
    all_dict["inputs"] = input
    all_dict["target"] = target
    all_dict["target_shape"] = target_shape
    all_dict["mean_lat_weight"] = mean_lat_weight
    all_dict["lat"] = lat
    all_dict["lon"] = lon

    input = input.to("cuda")
    # First split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        input, target, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed
    )

    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    train_dict["inputs"] = X_train
    train_dict["target"] = y_train
    train_dict["target_shape"] = y_train.shape
    train_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")
    train_dict["lat"] = lat.to("cuda")
    train_dict["lon"] = lon.to("cuda")
    train_dict["scaler"] = scaler

    val_dict["inputs"] = X_val
    val_dict["target"] = y_val
    val_dict["target_shape"] = y_val.shape
    val_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")
    val_dict["lat"] = lat.to("cuda")
    val_dict["lon"] = lon.to("cuda")
    val_dict["scaler"] = scaler

    test_dict["inputs"] = X_test
    test_dict["target"] = y_test
    test_dict["target_shape"] = y_test.shape
    test_dict["mean_lat_weight"] = mean_lat_weight.to("cuda")
    test_dict["lat"] = lat.to("cuda")
    test_dict["lon"] = lon.to("cuda")
    test_dict["scaler"] = scaler

    all_dict["scaler"] = scaler

    return train_dict, val_dict, test_dict, all_dict


def cmb_regression_preprocess(
    test_size, seed, dataset_dir, normalize, zscore_normalize=False, **kwargs
):
    target = torch.from_numpy(np.load(dataset_dir + "inp_data.npy")).float()

    npix = target.shape[0]
    theta, phi = hp.pix2ang(hp.npix2nside(target.shape[0]), torch.arange(npix))
    theta, phi = theta.to(torch.float32), phi.to(torch.float32)

    input = torch.column_stack((theta, phi))

    all_dict = dict()
    all_dict["inputs"] = input
    all_dict["target"] = target
    all_dict["target_shape"] = target.shape

    input = input.to("cuda")
    target = target.to("cuda")

    if normalize:
        scaler = MinMaxScalerTorch()
    elif zscore_normalize:
        scaler = StandardScalerTorch()
    else:
        raise Exception(
            "Set normalize argument properly (normalize or zscore_normalize"
        )

    scaler.fit(target)
    target = scaler.transform(target)

    X_train, X_test, y_train, y_test = train_test_split(
        input, target, test_size=0.2, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed
    )

    y_train, y_val, y_test = (
        y_train.unsqueeze(-1),
        y_val.unsqueeze(-1),
        y_test.unsqueeze(-1),
    )

    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    train_dict["inputs"] = X_train
    train_dict["target"] = y_train
    train_dict["target_shape"] = y_train.shape
    train_dict["scaler"] = scaler

    val_dict["inputs"] = X_val
    val_dict["target"] = y_val
    val_dict["target_shape"] = y_val.shape
    val_dict["scaler"] = scaler

    test_dict["inputs"] = X_test
    test_dict["target"] = y_test
    test_dict["target_shape"] = y_test.shape
    test_dict["scaler"] = scaler

    all_dict["scaler"] = scaler

    return (
        train_dict,
        val_dict,
        test_dict,
        all_dict,
    )


def temporal_preprocess(
    seed,
    dataset_dir,
    output_dim,
    time_resolution,
    normalize,
    zscore_normalize=False,
    **kwargs
):
    filenames = get_filenames(dataset_dir)
    # Read file and store in variables
    with nc.Dataset(filenames, "r") as f:
        for variable in f.variables:
            if variable == "latitude":
                lat = (
                    torch.from_numpy(f.variables[variable][:]).float().to("cuda")
                ) 
            elif variable == "longitude":
                lon = (
                    torch.from_numpy(f.variables[variable][:]).float().to("cuda")
                ) 
            elif variable == "time":
                time = (
                    torch.from_numpy(f.variables[variable][:]).float().to("cuda")
                ) 
            else:
                target = (
                    torch.from_numpy(f.variables[variable][:]).float().to("cuda")
                ) 

    """Convert lat, lon from degree to radian"""
    lat = torch.deg2rad(lat)  
    lon = torch.deg2rad(lon)  

    """Time resolution sampling, 1 for hour, 24 for day, 168 for week"""
    time_resolution_index = torch.arange(0, len(time), time_resolution)[:30]
    time = time[time_resolution_index]
    target = target[time_resolution_index]

    """Time normalization: max_dim [8760], normalize to [0,1]"""
    norm_time = (time - time.min()) / (time.max() - time.min())

    mean_lat_weight = torch.abs(torch.cos(lat)).mean().float()  

    if normalize:
        scaler = MinMaxScalerTorch()
    elif zscore_normalize:
        scaler = StandardScalerTorch()
    else:
        raise Exception(
            "Set normalize argument properly (normalize or zscore_normalize"
        )

    scaler.fit(target.flatten())
    target = scaler.transform(target)

    start = [0, 1]
    step = [2, 2]

    train_dict = dict()
    val_dict = dict()
    test_dict = dict()

    for i in range(len(start)):
        time_idx = torch.arange(start[i], len(norm_time), step[i])
        _time = norm_time[time_idx]
        _target = target[time_idx]
        _target_shape = _target.shape

        _time, _lat, _lon = torch.meshgrid(_time, lat, lon)

        # Split train and validation dataset
        if start[i] == 0:
            train_input = []
            train_target = []

            val_input = []
            val_target = []

            for j in range(len(_time)):
                _input = torch.stack(
                    [_lat[j].flatten(), _lon[j].flatten(), _time[j].flatten()], dim=-1
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    _input, _target[j].flatten(), test_size=0.2, random_state=seed
                )
                train_input.append(X_train)
                train_target.append(y_train)
                val_input.append(X_val)
                val_target.append(y_val)

            train_input = torch.stack(train_input)
            train_target = torch.stack(train_target)
            val_input = torch.stack(val_input)
            val_target = torch.stack(val_target)

            train_dict["inputs"] = train_input.reshape(-1, 3)
            train_dict["target"] = train_target.reshape(-1, output_dim)
            train_dict["mean_lat_weight"] = mean_lat_weight
            train_dict["target_shape"] = train_target.shape
            train_dict["scaler"] = scaler

            val_dict["inputs"] = val_input.reshape(-1, 3)
            val_dict["target"] = val_target.reshape(-1, output_dim)
            val_dict["mean_lat_weight"] = mean_lat_weight
            val_dict["target_shape"] = val_target.shape
            val_dict["scaler"] = scaler

        else:
            _lat, _lon, _time = _lat.flatten(), _lon.flatten(), _time.flatten()
            _target = _target.reshape(-1, output_dim)

            _inputs = torch.stack([_lat, _lon, _time], dim=-1)
            test_dict["inputs"] = _inputs
            test_dict["target"] = _target
            test_dict["mean_lat_weight"] = mean_lat_weight
            test_dict["target_shape"] = _target.shape
            test_dict["scaler"] = scaler

    return train_dict, val_dict, test_dict


def get_filenames(dataset_dir):
    filenames = glob.glob(os.path.join(dataset_dir, "*.nc"))
    return sorted(filenames)[0]
