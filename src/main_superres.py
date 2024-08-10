from argparse import ArgumentParser
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.data import weather_sr_preprocess, cmb_sr_preprocess
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["WANDB__SERVICE_WAIT"] = "300"


from datasets import spatial  # , temporal, temporal_ginr, spatial_ginr
from models import healpix, equirectangular

model_dict = {"healpix": healpix, "equirectangular": equirectangular}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model", type=str, default="healpix")

    # Dataset argument
    parser.add_argument("--img_idx", type=int, default=2)
    parser.add_argument("--normalize", default=False, action="store_true")
    parser.add_argument("--zscore_normalize", default=False, action="store_true")
    parser.add_argument("--data_year", default=None)  # For weather temporal
    parser.add_argument("--time_resolution", type=int, default=24)
    parser.add_argument("--time_len", type=int, default=48)
    parser.add_argument("--downscale_factor", type=int, default=2)

    # Model argument
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=4)
    parser.add_argument("--skip", default=False, action="store_true")
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mapping_size", default=256, type=int)
    parser.add_argument("--n_levels", default=7, type=int)
    parser.add_argument("--levels", default=4, type=int)
    parser.add_argument("--n_features_per_level", default=2, type=int)
    parser.add_argument("--resolution", default=2, type=int)
    parser.add_argument("--input_dim", default=3, type=int)
    parser.add_argument("--init_a", default=0.0, type=float)
    parser.add_argument("--init_b", default=0.01, type=float)
    parser.add_argument("--base_resol", default=16, type=int)
    parser.add_argument("--great_circle", default=False, action="store_true")
    parser.add_argument("--pole_singularity", default=False, action="store_true")
    parser.add_argument("--east_west", default=False, action="store_true")
    parser.add_argument("--upscale_factor", default=2.0, type=float)

    # Learning argument
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--lr_patience", type=int, default=1000)
    parser.add_argument(
        "--task", type=str, default="sr", choices=["reg", "sr", "temporal"]
    )

    parser.add_argument("--project_name", type=str, default="test")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    pl.seed_everything(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # input_dim = args.input_dim

    args.time = False
    args.input_dim = args.input_dim + (1 if args.time else 0)
    args.output_dim = 3 if "360" in args.dataset_dir else 1
    # Log
    logger = WandbLogger(
        config=args,
        name=args.model,
        project=args.project_name,
    )

    if "cmb" in args.dataset_dir:
        # train_dataset = train_dataset.detach().cpu()
        scaler, train_dict, val_dict, all_dict = cmb_sr_preprocess(
            seed=args.seed,
            input_dim=args.input_dim,
            dataset_dir=args.dataset_dir,
            downscale_factor=args.downscale_factor,
            normalize=args.normalize,
            zscore_normalize=args.zscore_normalize,
        )

    elif "spatial" in args.dataset_dir:
        scaler, train_dict, val_dict, all_dict = weather_sr_preprocess(
            seed=args.seed,
            input_dim=args.input_dim,
            dataset_dir=args.dataset_dir,
            downscale_factor=args.downscale_factor,
            normalize=args.normalize,
            zscore_normalize=args.zscore_normalize,
        )

    train_dataset = spatial.Dataset(data_dict=train_dict)
    val_dataset = spatial.Dataset(data_dict=val_dict)
    all_dataset = spatial.Dataset(data_dict=all_dict)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        all_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # Model
    if "cmb" in args.dataset_dir:
        model = model_dict[args.model].INP_INR(
            all_dataset=all_dataset, **vars(args)
        ) 
    elif "spatial" in args.dataset_dir:
        model = model_dict[args.model].INR(all_dataset=all_dataset, **vars(args))

    # Pass scaler from dataset to model
    model.scaler = scaler
    model.normalize = args.zscore_normalize or args.normalize
    if "spatial" in args.dataset_dir:
        model.all_dataset = all_dataset

    # Learning
    lrmonitor_cb = LearningRateMonitor(logging_interval="step")

    checkpoint_cb = ModelCheckpoint(
        monitor="valid_psnr",
        mode="max",
        filename="best-{epoch}-{best_valid_psnr:.2f}",
        save_top_k=1,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=args.max_epochs,
        log_every_n_steps=1,
        callbacks=[lrmonitor_cb, checkpoint_cb],
        logger=logger,
        gpus=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else None,
        # fast_dev_run=True,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    best_model_path = checkpoint_cb.best_model_path
    best_model_path = best_model_path.strip("'")

    if os.path.exists(best_model_path):
        new_best_model_path = best_model_path.replace("'", "")
        os.rename(best_model_path, new_best_model_path)

    if best_model_path:
        best_model = model.load_from_checkpoint(
            best_model_path, scaler=scaler, all_dataset=all_dataset, **vars(args)
        )
        best_model.scaler = scaler
        best_model.normalize = args.zscore_normalize or args.normalize
        if "spatial" in args.dataset_dir:
            best_model.all_dataset = all_dataset
    else:
        raise Exception("Best model is not saved properly")
