import torch
import pytorch_lightning as pl
from utils.utils import to_cartesian, mse2psnr, col_to_cartesian
from torch.optim import lr_scheduler
import copy


class MODEL(pl.LightningModule):
    def __init__(
        self,
        lr,
        lr_patience,
        model,
        normalize,
        zscore_normalize,
        all_dataset,
        time,
        input_dim,
        task,
        **kwargs
    ):
        super().__init__()

        self.lr = lr
        self.task = task
        self.lr_patience = lr_patience
        self.normalize = normalize or zscore_normalize
        self.scaler = None
        self.target_normalize = False
        self.model = model
        self._device = None
        self.all_dataset = all_dataset
        self.last_full_train_psnr = None
        self.last_full_train_rmse = None
        self.best_full_train_psnr = 0
        self.best_valid_w_psnr = 0
        self.best_valid_w_mse = 10000
        self.best_valid_w_rmse = 10000
        self.time = time
        self.input_dim = input_dim

    def metric_all(self, device, mode="train"):
        with torch.no_grad():
            all_data = copy.deepcopy(self.all_dataset[:])

            all_inputs, all_target = all_data["inputs"].to(device), all_data[
                "target"
            ].to(device)
            mean_lat_weight = all_data["mean_lat_weight"].to(device)

            if self.time:
                if self.input_dim == 4:
                    proceed_inputs = copy.deepcopy(
                        torch.cat(
                            (to_cartesian(all_inputs[..., :2]), all_inputs[..., 2:]),
                            dim=-1,
                        )
                    )
                    lat = all_inputs[..., :1]

                elif self.input_dim == 3:
                    proceed_inputs = copy.deepcopy(all_inputs)
                    lat = torch.deg2rad(all_inputs[..., :1])

            else:
                if self.input_dim == 3:
                    proceed_inputs = copy.deepcopy(to_cartesian(all_inputs))
                    lat = all_inputs[..., :1]

                else:
                    proceed_inputs = copy.deepcopy(all_inputs)
                    lat = torch.deg2rad(all_inputs[..., :1])

            all_pred = self(proceed_inputs)

            weights = torch.abs(torch.cos(lat))
            weights = weights / mean_lat_weight
            if weights.shape[-1] == 1:
                weights = weights.squeeze(-1)

            error = torch.sum((all_pred - all_target) ** 2, dim=-1)
            error = weights * error
            all_loss = error.mean()
            all_rmse = torch.sqrt(all_loss).item()
            all_w_psnr_val = mse2psnr(all_loss)

            if mode == "train" and self.best_full_train_psnr < all_w_psnr_val:
                self.best_full_train_psnr = all_w_psnr_val

            self.log(
                "full_" + mode + "_rmse",
                all_rmse,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )
            self.log(
                "full_" + mode + "_psnr",
                all_w_psnr_val,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data["mean_lat_weight"]
        if mean_lat_weight.shape[0] > 1:
            mean_lat_weight = mean_lat_weight[0]

        if self.input_dim == 2:
            rad = torch.deg2rad(inputs)
            rad_lat = rad[..., :1]

            weights = torch.abs(torch.cos(rad_lat))
        else:
            weights = torch.abs(torch.cos(inputs[..., :1]))
        weights = weights / mean_lat_weight

        if self.time:
            if self.input_dim == 4:
                inputs = torch.cat(
                    (to_cartesian(inputs[..., :2]), inputs[..., 2:]), dim=-1
                )

        else:
            if self.input_dim == 3:
                inputs = to_cartesian(inputs)

        pred = self.forward(inputs)
        self._device = pred.device

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error

        loss = error.mean()

        rmse = torch.sqrt(loss)
        w_psnr_val = mse2psnr(loss)

        self.log("train_mse", loss, prog_bar=True, sync_dist=True)
        self.log("train_psnr", w_psnr_val, prog_bar=True, sync_dist=True)
        self.log("train_rmse", rmse, prog_bar=True, sync_dist=True)

        ############################# Unnormalized RMSE and MSE ##############################
        pred = self.scaler.inverse_transform(pred)
        target = self.scaler.inverse_transform(target)
        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error
        error = error.mean()

        self.log("unnorm_train_mse", error, prog_bar=True, sync_dist=True)
        self.log("unnorm_train_rmse", torch.sqrt(error), prog_bar=True, sync_dist=True)
        ######################################################################################

        return {"loss": loss, "train_psnr": w_psnr_val.item()}

    def on_train_epoch_end(self):
        if (
            not self.time
            and self.task != "reg"
            and self.task != "sr"
        ): 
            self.eval()
            with torch.no_grad():
                self.metric_all(device=self._device, mode="train")
            if self.current_epoch == self.trainer.max_epochs - 1:
                self.log(
                    "best_full_train_psnr",
                    self.best_full_train_psnr,
                    prog_bar=True,
                    sync_dist=True,
                )

    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data["mean_lat_weight"]
        if mean_lat_weight.shape[0] > 1:
            mean_lat_weight = mean_lat_weight[0]

        if self.input_dim == 2:
            rad = torch.deg2rad(inputs)
            rad_lat = rad[..., :1]

            weights = torch.abs(torch.cos(rad_lat))
        else:
            weights = torch.abs(torch.cos(inputs[..., :1]))
        weights = weights / mean_lat_weight

        if self.time:
            if self.input_dim == 4:
                inputs = torch.cat(
                    (to_cartesian(inputs[..., :2]), inputs[..., 2:]), dim=-1
                )

        else:
            if self.input_dim == 3:
                inputs = to_cartesian(inputs)

        pred = self.forward(inputs)
        self._device = pred.device

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error

        loss = error.mean()

        rmse = torch.sqrt(loss)
        w_psnr_val = mse2psnr(loss)
        if w_psnr_val > self.best_valid_w_psnr:
            self.best_valid_w_psnr = w_psnr_val
        if loss < self.best_valid_w_mse:
            self.best_valid_w_mse = loss
        if rmse < self.best_valid_w_rmse:
            self.best_valid_w_rmse = rmse

        self.log("valid_mse", loss, prog_bar=True, sync_dist=True)
        self.log("valid_psnr", w_psnr_val, prog_bar=True, sync_dist=True)
        self.log("valid_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log(
            "best_valid_psnr", self.best_valid_w_psnr, prog_bar=True, sync_dist=True
        )
        self.log(
            "best_valid_rmse", self.best_valid_w_rmse, prog_bar=True, sync_dist=True
        )
        self.log("best_valid_mse", self.best_valid_w_mse, prog_bar=True, sync_dist=True)

        ############################# Unnormalized RMSE and MSE ##############################
        pred = self.scaler.inverse_transform(pred)
        target = self.scaler.inverse_transform(target)
        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error
        error = error.mean()

        self.log("unnorm_valid_mse", error, prog_bar=True, sync_dist=True)
        self.log("unnorm_valid_rmse", torch.sqrt(error), prog_bar=True, sync_dist=True)
        ######################################################################################

        return {"loss": loss, "valid_psnr": w_psnr_val.item()}

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        mean_lat_weight = data["mean_lat_weight"]
        if mean_lat_weight.shape[0] > 1:
            mean_lat_weight = mean_lat_weight[0]

        if self.input_dim == 2:
            rad = torch.deg2rad(inputs)
            rad_lat = rad[..., :1]

            weights = torch.abs(torch.cos(rad_lat))
        else:
            weights = torch.abs(torch.cos(inputs[..., :1]))
        weights = weights / mean_lat_weight

        if self.time:
            if self.input_dim == 4:
                inputs = torch.cat(
                    (to_cartesian(inputs[..., :2]), inputs[..., 2:]), dim=-1
                )

        else:
            if self.input_dim == 3:
                inputs = to_cartesian(inputs)

        pred = self.forward(inputs) 
        self._device = pred.device

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error

        loss = error.mean()

        rmse = torch.sqrt(loss)
        w_psnr_val = mse2psnr(loss)

        self.log("test_mse", loss, prog_bar=True, sync_dist=True)
        self.log("test_psnr", w_psnr_val, prog_bar=True, sync_dist=True)
        self.log("test_rmse", rmse, prog_bar=True, sync_dist=True)
        ############################# Unnormalized RMSE and MSE ##############################
        pred = self.scaler.inverse_transform(pred)
        target = self.scaler.inverse_transform(target)
        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        if len(error.shape) > len(weights.shape):
            error = error.squeeze(-1)
        error = weights * error
        error = error.mean()

        self.log("unnorm_test_mse", error, prog_bar=True, sync_dist=True)
        self.log("unnorm_test_rmse", torch.sqrt(error), prog_bar=True, sync_dist=True)
        ######################################################################################

        return {"loss": loss, "valid_psnr": w_psnr_val.item()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {
            "scheduler": scheduler,
            "monitor": "train_rmse",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}


class INP_MODEL(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        lr,
        lr_patience,
        model,
        normalize,
        zscore_normalize,
        time,
        task,
        **kwargs
    ):
        super().__init__()

        self.lr = lr
        self.time = time
        self.lr_patience = lr_patience
        self.normalize = normalize or zscore_normalize
        self.scaler = None
        self.target_normalize = False
        self.model = model
        self._device = None
        self.best_train_psnr = 0
        self.best_train_mse = 10000
        self.best_train_rmse = 10000
        self.best_valid_psnr = 0
        self.best_valid_mse = 10000
        self.best_valid_rmse = 10000
        self.best_full_train_psnr = 0
        self.input_dim = input_dim
        self.task = task

    def metric_all(self, device, mode="train"):
        with torch.no_grad():
            all_data = copy.deepcopy(self.all_dataset[:])

            all_inputs, all_target = all_data["inputs"].to(device), all_data[
                "target"
            ].to(device)

            if self.input_dim == 3:
                proceed_inputs = copy.deepcopy(to_cartesian(all_inputs))

            else:
                proceed_inputs = copy.deepcopy(all_inputs)

            all_pred = self(proceed_inputs)

            error = torch.sum((all_pred - all_target) ** 2, dim=-1)
            all_loss = error.mean()
            all_rmse = torch.sqrt(all_loss).item()
            all_psnr_val = mse2psnr(all_loss)

            if mode == "train" and self.best_full_train_psnr < all_psnr_val:
                self.best_full_train_psnr = all_psnr_val

            self.log(
                "full_" + mode + "_rmse",
                all_rmse,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )
            self.log(
                "full_" + mode + "_psnr",
                all_psnr_val,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )

    def training_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        if self.input_dim == 3:
            inputs = col_to_cartesian(inputs)

        pred = self.forward(inputs)
        self._device = pred.device

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)

        loss = error.mean()
        rmse = torch.sqrt(loss)
        psnr_val = mse2psnr(loss)

        if psnr_val > self.best_train_psnr:
            self.best_train_psnr = psnr_val
        if loss < self.best_train_mse:
            self.best_train_mse = loss
        if rmse < self.best_train_rmse:
            self.best_train_rmse = rmse

        self.log("train_mse", loss, prog_bar=True, sync_dist=True)
        self.log("train_psnr", psnr_val, prog_bar=True, sync_dist=True)
        self.log("train_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("best_train_psnr", self.best_train_psnr, prog_bar=True, sync_dist=True)
        self.log("best_train_rmse", self.best_train_rmse, prog_bar=True, sync_dist=True)
        self.log("best_train_mse", self.best_train_mse, prog_bar=True, sync_dist=True)

        ############################# Unnormalized RMSE and MSE ##############################
        pred = self.scaler.inverse_transform(pred)
        target = self.scaler.inverse_transform(target)

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error = error.mean()

        self.log("unnorm_train_mse", error, prog_bar=True, sync_dist=True)
        self.log("unnorm_train_rmse", torch.sqrt(error), prog_bar=True, sync_dist=True)
        ######################################################################################

        return {"loss": loss, "train_psnr": psnr_val.item()}

    def validation_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        if self.input_dim == 3:
            inputs = col_to_cartesian(inputs)
        pred = self.forward(inputs)
        self._device = pred.device

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)

        loss = error.mean()
        rmse = torch.sqrt(loss)
        psnr_val = mse2psnr(loss)

        if psnr_val > self.best_valid_psnr:
            self.best_valid_psnr = psnr_val
        if loss < self.best_valid_mse:
            self.best_valid_mse = loss
        if rmse < self.best_valid_rmse:
            self.best_valid_rmse = rmse

        self.log("valid_mse", loss, prog_bar=True, sync_dist=True)
        self.log("valid_psnr", psnr_val, prog_bar=True, sync_dist=True)
        self.log("valid_rmse", rmse, prog_bar=True, sync_dist=True)
        self.log("best_valid_psnr", self.best_valid_psnr, prog_bar=True, sync_dist=True)
        self.log("best_valid_rmse", self.best_valid_rmse, prog_bar=True, sync_dist=True)
        self.log("best_valid_mse", self.best_valid_mse, prog_bar=True, sync_dist=True)

        ############################# Unnormalized RMSE and MSE ##############################
        pred = self.scaler.inverse_transform(pred)
        target = self.scaler.inverse_transform(target)
        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error = error.mean()

        self.log("unnorm_valid_mse", error, prog_bar=True, sync_dist=True)
        self.log("unnorm_valid_rmse", torch.sqrt(error), prog_bar=True, sync_dist=True)
        ######################################################################################

        return {"loss": loss, "valid_psnr": psnr_val.item()}

    def test_step(self, data, batch_idx):
        inputs, target = data["inputs"], data["target"]
        if self.input_dim == 3:
            inputs = col_to_cartesian(inputs)
        pred = self.forward(inputs)
        self._device = pred.device

        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)

        loss = error.mean()
        rmse = torch.sqrt(loss)
        psnr_val = mse2psnr(loss)
        self.log("test_mse", loss, prog_bar=True, sync_dist=True)
        self.log("test_psnr", psnr_val, prog_bar=True, sync_dist=True)
        self.log("test_rmse", rmse, prog_bar=True, sync_dist=True)

        ############################# Unnormalized RMSE and MSE ##############################
        pred = self.scaler.inverse_transform(pred)
        target = self.scaler.inverse_transform(target)
        error = torch.sum((pred - target) ** 2, dim=-1, keepdim=True)
        error = error.mean()

        self.log("unnorm_test_mse", error, prog_bar=True, sync_dist=True)
        self.log("unnorm_test_rmse", torch.sqrt(error), prog_bar=True, sync_dist=True)
        ######################################################################################
        return {"loss": loss, "test_psnr": psnr_val.item()}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.lr_patience, verbose=True
        )

        sch_dict = {
            "scheduler": scheduler,
            "monitor": "train_rmse",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": sch_dict}
