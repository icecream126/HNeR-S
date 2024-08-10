# Hybrid Neural Representations for Spherical Data
This is an official implementation of the paper [**Hybrid Neural Representations for Spherical Data (HNeR-S)** ](https://openreview.net/pdf/b41d07dfae94bd219ce05afe1814a0ecafaebfa7.pdf)accepted at ICML 2024.

# Dataset
Dataset preparation code (```src/datasets/generation/download.py```) is originally from [NNCompression github](https://github.com/spcl/NNCompression/blob/master/WeatherBench/src/download.py). 

Before, running the below code, one shoudl fill out the ```KEY``` value in ```download.py```. The value can be obtained by following the process at [CDS API website](https://cds.climate.copernicus.eu/api-how-to).
## Spatial dataset
Run the following codes to download weather datasets.
```
# Resolution 0.25
python src/datasets/generation/download.py --variable=geopotential --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_geopotential
python src/datasets/generation/download.py --variable=temperature --mode=single --level_type=pressure --years=2000 --resolution=0.25 --month=01 --day=01 --time=00:00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/spatial_0_25/era5_temperature
```

## Temporal dataset
Run the following codes to download weather datasets.
```
# Geopotential
python src/datasets/generation/download.py --variable=geopotential --mode=separate --level_type=pressure --years=2000 --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal/era5_geopotential

# Temperature
python src/datasets/generation/download.py --variable=temperature --mode=separate --level_type=pressure --years=2000 --resolution=1.00 --pressure_level=500 --custom_fn=data.nc --output_dir=dataset/temporal/era5_temperature
```


# How to run


## Arguments
* **dataset_dir** (str) 
* **downscale_factor** (int)
* **seed** (int)
* **model** (str) ```[healpix, equirect]```

## Example script

```
# Model : HEALPix
# Task : Spatial super resolution
# Dataset : Geopotential
# Downscale factor : x2
# Seed : 0

CUDA_VISIBLE_DEVICES=4 python src/main_superres.py \
    --dataset_dir dataset/spatial_0_25/era5_geopotential \
    --downscale_factor 2 \
    --seed 0 \
    --n_levels 9 \
    --n_features_per_level 2 \
    --input_dim 2 \
    --batch_size 4096 \
    --model healpix \
    --normalize \
    --skip 
```
