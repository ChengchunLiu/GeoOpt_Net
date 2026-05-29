import os
import numpy as np
import time
import json
import csv
from tqdm import tqdm

import torch
import torch.optim as optim

from GeoLoss import GeoLoss
from dataclasses import asdict

FLOAT_PRECISION = 16
PREC_TIME = 2
PREC_LR = 6
PREC_LONG = 16


def config_to_serializable_dict(config):
    cfg_dict = asdict(config)
    cfg_dict["data"]["device"] = str(config.data.device)
    return cfg_dict


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.config = config
        self.model = model.to(config.data.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=10,
        )

        loss_weights = {
            "rmsd": config.loss.rmsd,
            "bond": config.loss.bond,
            "angle": config.loss.angle,
            "dihedral": config.loss.dihedral,
        }
        self.loss_fn = GeoLoss(loss_weights)

        self.best_rmsd = float("inf")
        self.best_epoch = None

        self.param_count = sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )

        self.start_time_str = time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(
            config.data.save_dir,
            self.start_time_str,
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.metrics_csv = os.path.join(self.checkpoint_dir, "metrics.csv")

        with open(self.metrics_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Config"])
            config_dict = config_to_serializable_dict(config)
            writer.writerow([json.dumps(config_dict, indent=4)])

            writer.writerow([])
            writer.writerow([f"Model Parameter Count: {self.param_count}"])
            writer.writerow([])
            writer.writerow([
                "Epoch",
                "Train_Total",
                "Train_RMSD",
                "Train_Bond",
                "Train_BondRange",
                "Train_Angle",
                "Train_Dihedral",
                "Train_Batches",
                "Train_Samples",
                "Train_Time(s)",
                "Avg_Grad_Norm",
                "Val_Total",
                "Val_RMSD",
                "Val_Bond",
                "Val_BondRange",
                "Val_Angle",
                "Val_Dihedral",
                "Val_Batches",
                "Val_Samples",
                "Val_Time(s)",
                "LR",
                "Weight_Mean",
                "Weight_Std",
                "Best_Epoch",
            ])

    def make_model_input(self, batch):
        model_input = batch.clone()

        forbidden_keys = [
            "dft_pos",
            "target_pos",
            "ref_pos",
            "reference_pos",
            "optimized_pos",
            "dft_energy",
            "dft_forces",
            "y",
        ]

        for key in forbidden_keys:
            self._safe_delete_attr(model_input, key)

        return model_input

    @staticmethod
    def _safe_delete_attr(data, key):
        try:
            if key in data.keys():
                del data[key]
                return
        except Exception:
            pass

        try:
            if hasattr(data, key):
                delattr(data, key)
                return
        except Exception:
            pass

        try:
            for store in data.stores:
                if key in store:
                    del store[key]
        except Exception:
            pass

    def compute_grad_norm(self):
        total_norm, count = 0.0, 0

        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.detach().data.norm(2).item() ** 2
                count += 1

        total_norm = total_norm ** 0.5
        return total_norm / count if count > 0 else 0.0

    def get_weight_stats(self):
        params = [
            p.detach().cpu().numpy().ravel()
            for p in self.model.parameters()
            if p.requires_grad
        ]

        if not params:
            return 0.0, 0.0

        all_params = np.concatenate(params)
        return float(np.mean(all_params)), float(np.std(all_params))

    def _get_zero_tensor_like_loss(self, loss_dict, key="total"):
        if key in loss_dict and torch.is_tensor(loss_dict[key]):
            return loss_dict[key].detach().new_tensor(0.0)

        return torch.tensor(0.0, device=self.config.data.device)

    def train_epoch(self):
        self.model.train()

        total_loss_sum = 0.0
        total_rmsd_sum = 0.0
        total_bond_sum = 0.0
        total_bond_range_sum = 0.0
        total_angle_sum = 0.0
        total_dihedral_sum = 0.0
        grad_norm_sum = 0.0

        batch_count = 0
        sample_count = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        fmt = f".{FLOAT_PRECISION}f"

        for batch in pbar:
            batch = batch.to(self.config.data.device)
            sample_count += getattr(batch, "num_graphs", batch.x.size(0))

            self.optimizer.zero_grad()

            model_input = self.make_model_input(batch)

            pred = self.model(model_input)
            loss_dict = self.loss_fn(pred, batch)

            if torch.isnan(loss_dict["total"]):
                print("NaN detected in loss:", loss_dict)
                continue

            loss_dict["total"].backward()

            grad_norm = self.compute_grad_norm()
            grad_norm_sum += grad_norm

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bond_range_value = loss_dict.get(
                "bond_range",
                self._get_zero_tensor_like_loss(loss_dict),
            )

            total_loss_sum += loss_dict["total"].item()
            total_rmsd_sum += loss_dict["rmsd"].item()
            total_bond_sum += loss_dict["bond"].item()
            total_bond_range_sum += bond_range_value.item()
            total_angle_sum += loss_dict["angle"].item()
            total_dihedral_sum += loss_dict["dihedral"].item()
            batch_count += 1

            pbar.set_postfix({
                "total": format(loss_dict["total"].item(), fmt),
                "rmsd": format(loss_dict["rmsd"].item(), fmt),
                "bond": format(loss_dict["bond"].item(), fmt),
                "bond_range": format(bond_range_value.item(), fmt),
                "angle": format(loss_dict["angle"].item(), fmt),
                "dihedral": format(loss_dict["dihedral"].item(), fmt),
                "grad_norm": format(grad_norm, fmt),
            })

        train_time = time.time() - start_time

        avg_metrics = {
            "total": total_loss_sum / batch_count if batch_count > 0 else 0.0,
            "rmsd": total_rmsd_sum / batch_count if batch_count > 0 else 0.0,
            "bond": total_bond_sum / batch_count if batch_count > 0 else 0.0,
            "bond_range": total_bond_range_sum / batch_count if batch_count > 0 else 0.0,
            "angle": total_angle_sum / batch_count if batch_count > 0 else 0.0,
            "dihedral": total_dihedral_sum / batch_count if batch_count > 0 else 0.0,
            "batches": batch_count,
            "samples": sample_count,
            "time": train_time,
            "avg_grad_norm": grad_norm_sum / batch_count if batch_count > 0 else 0.0,
        }

        return avg_metrics

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        total_loss_sum = 0.0
        total_rmsd_sum = 0.0
        total_bond_sum = 0.0
        total_bond_range_sum = 0.0
        total_angle_sum = 0.0
        total_dihedral_sum = 0.0

        batch_count = 0
        sample_count = 0
        start_time = time.time()

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)

        fmt = f".{FLOAT_PRECISION}f"

        for batch in pbar:
            batch = batch.to(self.config.data.device)
            sample_count += getattr(batch, "num_graphs", batch.x.size(0))

            model_input = self.make_model_input(batch)

            pred = self.model(model_input)
            loss_dict = self.loss_fn(pred, batch)

            bond_range_value = loss_dict.get(
                "bond_range",
                self._get_zero_tensor_like_loss(loss_dict),
            )

            total_loss_sum += loss_dict["total"].item()
            total_rmsd_sum += loss_dict["rmsd"].item()
            total_bond_sum += loss_dict["bond"].item()
            total_bond_range_sum += bond_range_value.item()
            total_angle_sum += loss_dict["angle"].item()
            total_dihedral_sum += loss_dict["dihedral"].item()
            batch_count += 1

            pbar.set_postfix({
                "total": format(loss_dict["total"].item(), fmt),
                "rmsd": format(loss_dict["rmsd"].item(), fmt),
                "bond": format(loss_dict["bond"].item(), fmt),
                "bond_range": format(bond_range_value.item(), fmt),
                "angle": format(loss_dict["angle"].item(), fmt),
                "dihedral": format(loss_dict["dihedral"].item(), fmt),
            })

        val_time = time.time() - start_time

        avg_metrics = {
            "total": total_loss_sum / batch_count if batch_count > 0 else 0.0,
            "rmsd": total_rmsd_sum / batch_count if batch_count > 0 else 0.0,
            "bond": total_bond_sum / batch_count if batch_count > 0 else 0.0,
            "bond_range": total_bond_range_sum / batch_count if batch_count > 0 else 0.0,
            "angle": total_angle_sum / batch_count if batch_count > 0 else 0.0,
            "dihedral": total_dihedral_sum / batch_count if batch_count > 0 else 0.0,
            "batches": batch_count,
            "samples": sample_count,
            "time": val_time,
        }

        return avg_metrics

    def train(self):
        fmt_time = f".{PREC_TIME}f"
        fmt_lr = f".{PREC_LR}f"
        fmt_long = f".{PREC_LONG}f"

        for epoch in range(1, self.config.train.num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["total"])

            weight_mean, weight_std = self.get_weight_stats()

            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_dir, f"model_epoch{epoch:03d}.pt"),
            )

            if val_metrics["rmsd"] < self.best_rmsd:
                self.best_rmsd = val_metrics["rmsd"]
                self.best_epoch = epoch

                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_model.pt"),
                )

                best_str = f" (Best Epoch: {epoch})"
            else:
                best_str = f" (Best Epoch: {self.best_epoch})"

            print(
                f"Epoch {epoch:03d} | "
                f"Train - Total: {format(train_metrics['total'], fmt_long)}, "
                f"RMSD: {format(train_metrics['rmsd'], fmt_long)}, "
                f"Val - Total: {format(val_metrics['total'], fmt_long)}, "
                f"RMSD: {format(val_metrics['rmsd'], fmt_long)}, "
                f"LR: {format(current_lr, fmt_lr)}, "
                f"Weights (mean,std): "
                f"({format(weight_mean, fmt_long)},{format(weight_std, fmt_long)})"
                f"{best_str}"
            )

            with open(self.metrics_csv, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)

                writer.writerow([
                    epoch,
                    format(train_metrics["total"], fmt_long),
                    format(train_metrics["rmsd"], fmt_long),
                    format(train_metrics["bond"], fmt_long),
                    format(train_metrics["bond_range"], fmt_long),
                    format(train_metrics["angle"], fmt_long),
                    format(train_metrics["dihedral"], fmt_long),
                    train_metrics["batches"],
                    train_metrics["samples"],
                    format(train_metrics["time"], fmt_time),
                    format(train_metrics["avg_grad_norm"], fmt_long),
                    format(val_metrics["total"], fmt_long),
                    format(val_metrics["rmsd"], fmt_long),
                    format(val_metrics["bond"], fmt_long),
                    format(val_metrics["bond_range"], fmt_long),
                    format(val_metrics["angle"], fmt_long),
                    format(val_metrics["dihedral"], fmt_long),
                    val_metrics["batches"],
                    val_metrics["samples"],
                    format(val_metrics["time"], fmt_time),
                    format(current_lr, fmt_lr),
                    format(weight_mean, fmt_long),
                    format(weight_std, fmt_long),
                    self.best_epoch if self.best_epoch is not None else "NA",
                ])