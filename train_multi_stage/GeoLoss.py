# GeoLoss.py
import torch
import torch.nn as nn


class GeoLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, pred, data):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        target = torch.nan_to_num(data.dft_pos, nan=0.0, posinf=1e6, neginf=-1e6)

        aligned_pred, rmsd = self.kabsch_align(pred, target, batch=data.batch)

        bond_loss = torch.nan_to_num(self.bond_length_loss(aligned_pred, data), nan=0.0)
        angle_loss = torch.nan_to_num(self.bond_angle_loss(aligned_pred, data), nan=0.0)
        dihedral_loss = torch.nan_to_num(self.dihedral_angle_loss(aligned_pred, data), nan=0.0)

        bond_ranges = {
            (6, 8, 1.0): (1.324, 1.503),
            (6, 6, 1.0): (1.364, 1.610),
            (6, 8, 2.0): (1.181, 1.235),
            (1, 6, 1.0): (1.062, 1.117),
            (6, 7, 3.0): (1.153, 1.163),
            (6, 7, 1.0): (1.342, 1.575),
            (1, 7, 1.0): (1.003, 1.027),
            (6, 7, 2.0): (1.243, 1.299),
            (1, 8, 1.0): (0.961, 0.982),
            (6, 6, 2.0): (1.323, 1.371),
            (6, 6, 1.5): (1.329, 1.531),
            (6, 7, 1.5): (1.260, 1.445),
            (7, 7, 1.5): (1.203, 1.414),
            (6, 9, 1.0): (1.300, 1.360),
            (6, 8, 1.5): (1.299, 1.493),
            (7, 8, 1.0): (1.215, 1.427),
            (6, 6, 3.0): (1.199, 1.215),
            (7, 8, 1.5): (1.333, 1.612),
            (7, 8, 2.0): (1.214, 1.246),
        }
        bond_range_loss = torch.nan_to_num(
            self.bond_length_range_loss(aligned_pred, data, bond_ranges, weight=0.5),
            nan=0.0
        )

        total_loss = (
            self.weights["rmsd"] * rmsd +
            self.weights["bond"] * bond_loss +
            bond_range_loss +
            self.weights["angle"] * angle_loss +
            self.weights["dihedral"] * dihedral_loss
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0)

        return {
            "total": total_loss,
            "rmsd": rmsd,
            "bond": bond_loss,
            "bond_range": bond_range_loss,
            "angle": angle_loss,
            "dihedral": dihedral_loss,
        }

    # --------- Kabsch 对齐 ---------
    def kabsch_align_single(self, pred, target):
        if pred.shape[0] < 2:
            diff = pred - target
            rmsd = torch.sqrt(torch.mean(diff ** 2) + 1e-8)
            return pred, rmsd

        pred_mean = pred.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        pred_c = pred - pred_mean
        target_c = target - target_mean

        epsilon = 1e-6
        H = pred_c.t() @ target_c + epsilon
        try:
            U, S, Vt = torch.linalg.svd(H, full_matrices=False)
        except Exception:
            aligned = pred
            rmsd = torch.sqrt(torch.mean((pred - target) ** 2) + 1e-8)
            return aligned, rmsd

        R = Vt.T @ U.T
        if torch.linalg.det(R) < 0:
            Vt = Vt.clone()
            Vt[:, -1] = -Vt[:, -1]
            R = Vt.T @ U.T

        t = target_mean - pred_mean @ R.T
        aligned = pred @ R.T + t

        diff = aligned - target
        rmsd = torch.sqrt(torch.mean(diff ** 2) + 1e-8)
        return aligned, rmsd

    def kabsch_align(self, pred, target, batch=None):
        if batch is None:
            return self.kabsch_align_single(pred, target)

        unique_batches = torch.unique(batch)
        aligned_list, rmsd_list = [], []
        for b in unique_batches:
            mask = (batch == b)
            pred_i = pred[mask]
            target_i = target[mask]
            aligned_i, rmsd_i = self.kabsch_align_single(pred_i, target_i)
            aligned_list.append(aligned_i)
            rmsd_list.append(rmsd_i)

        aligned = torch.cat(aligned_list, dim=0)
        rmsd = torch.mean(torch.stack(rmsd_list))
        return aligned, rmsd

    # --------- 几何损失 ---------
    def bond_length_loss(self, pred_coords, data):
        indices = data.true_dist_indices.to(pred_coords.device)
        if indices.numel() == 0:
            return torch.tensor(0.0, device=pred_coords.device)

        pred_len = torch.norm(
            pred_coords[indices[:, 0]] - pred_coords[indices[:, 1]], dim=-1
        )
        true_len = data.true_dists.to(pred_coords.device)
        return nn.functional.mse_loss(pred_len, true_len)

    def bond_length_range_loss(self, pred_coords, data, bond_ranges, weight=1.0):
        indices = data.true_dist_indices.to(pred_coords.device)
        if indices.numel() == 0:
            return torch.tensor(0.0, device=pred_coords.device)

        pred_len = torch.norm(
            pred_coords[indices[:, 0]] - pred_coords[indices[:, 1]], dim=-1
        )

        atomic_numbers = data.atomic_numbers.to(pred_coords.device)
        a = atomic_numbers[indices[:, 0]]
        b = atomic_numbers[indices[:, 1]]
        a_sorted = torch.min(a, b)
        b_sorted = torch.max(a, b)

        # ========= 关键修改：稳健展开 true_bond_types =========
        tbt = data.true_bond_types

        if isinstance(tbt, torch.Tensor):
            # 已经是 1D tensor（例如你将来重处理数据时直接存成 tensor）
            bond_types = tbt.to(pred_coords.device).float()
        else:
            # 兼容当前情况：DataLoader 会把每个分子的 list 收集成 list of lists
            flat = []

            def _flatten(x):
                if isinstance(x, torch.Tensor):
                    flat.extend(x.view(-1).tolist())
                elif isinstance(x, (list, tuple)):
                    for v in x:
                        _flatten(v)
                else:
                    flat.append(float(x))

            _flatten(tbt)
            bond_types = torch.tensor(flat, dtype=torch.float, device=pred_coords.device)
        # ========= 关键修改结束 =========

        loss_list = []
        for key, (L_min, L_max) in bond_ranges.items():
            atom1, atom2, btype = key
            mask = (
                (a_sorted == atom1)
                & (b_sorted == atom2)
                & (torch.abs(bond_types - btype) < 1e-5)
            )
            if mask.sum() == 0:
                continue
            lengths = pred_len[mask]
            loss_below = torch.relu(L_min - lengths)
            loss_above = torch.relu(lengths - L_max)
            loss_key = torch.mean(loss_below ** 2 + loss_above ** 2)
            loss_list.append(loss_key)

        if loss_list:
            range_loss = torch.mean(torch.stack(loss_list))
        else:
            range_loss = torch.tensor(0.0, device=pred_coords.device)

        return weight * range_loss

    def bond_angle_loss(self, pred_coords, data):
        indices = data.true_angle_indices.to(pred_coords.device)
        if indices.numel() == 0:
            return torch.tensor(0.0, device=pred_coords.device)

        i, j, k = indices[:, 0], indices[:, 1], indices[:, 2]
        v1 = pred_coords[i] - pred_coords[j]
        v2 = pred_coords[k] - pred_coords[j]
        dot = (v1 * v2).sum(dim=-1)
        n1 = torch.norm(v1, dim=-1) + 1e-8
        n2 = torch.norm(v2, dim=-1) + 1e-8
        cosang = torch.clamp(dot / (n1 * n2), -1 + 1e-7, 1 - 1e-7)
        pred_ang = torch.acos(cosang)

        true_ang = data.true_angles.to(pred_coords.device)
        return nn.functional.mse_loss(pred_ang, true_ang)

    def dihedral_angle_loss(self, pred_coords, data):
        indices = data.true_dihedral_indices.to(pred_coords.device)
        if indices.numel() == 0:
            return torch.tensor(0.0, device=pred_coords.device)

        p0 = pred_coords[indices[:, 0]]
        p1 = pred_coords[indices[:, 1]]
        p2 = pred_coords[indices[:, 2]]
        p3 = pred_coords[indices[:, 3]]

        pred_dih = self.compute_dihedral(p0, p1, p2, p3)
        true_dih = data.true_dihedrals.to(pred_coords.device)

        diff = pred_dih - true_dih
        diff = torch.remainder(diff + torch.pi, 2 * torch.pi) - torch.pi
        return torch.mean(diff ** 2)

    def compute_dihedral(self, p0, p1, p2, p3):
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        b1_norm = b1 / (torch.norm(b1, dim=-1, keepdim=True) + 1e-6)
        v = b0 - torch.sum(b0 * b1_norm, dim=-1, keepdim=True) * b1_norm
        w = b2 - torch.sum(b2 * b1_norm, dim=-1, keepdim=True) * b1_norm
        x = torch.sum(v * w, dim=-1)
        y = torch.sum(torch.cross(b1_norm, v, dim=-1) * w, dim=-1)
        return SafeAtan2.apply(y, x, 1e-8)


class SafeAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x, eps=1e-8):
        ctx.eps = eps
        ctx.save_for_backward(y, x)
        denom = x ** 2 + y ** 2
        mask = (denom < eps)
        safe_x = torch.where(mask, torch.full_like(x, eps), x)
        safe_y = torch.where(mask, torch.zeros_like(y), y)
        return torch.atan2(safe_y, safe_x)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        y, x = ctx.saved_tensors
        denom = x ** 2 + y ** 2
        d = torch.clamp(denom, min=eps)
        grad_y = grad_output * (x / d)
        grad_x = grad_output * (-y / d)
        mask = (denom < eps)
        grad_y = torch.where(mask, torch.zeros_like(grad_y), grad_y)
        grad_x = torch.where(mask, torch.zeros_like(grad_x), grad_x)
        return grad_y, grad_x, None
