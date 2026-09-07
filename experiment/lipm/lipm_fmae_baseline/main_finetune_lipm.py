#!/usr/bin/env python3
"""Fine-tune the official LiPM backbone on FMAE downstream datasets."""

import argparse
import datetime
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
LIPM_ROOT = THIS_DIR.parent


def find_repo_root(start):
    for path in [start] + list(start.parents):
        if (path / "util" / "datasets.py").exists() and (path / "engine_finetune.py").exists():
            return path
    raise RuntimeError(f"Could not find FMAE repository root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIPM_ROOT))

from LiPM.battery_model import BatteryNet  # noqa: E402
from util.datasets import PreprocessNormalizer, load_dataset  # noqa: E402
import util.evaluation  # noqa: E402
from engine_finetune import evaluate_RUL  # noqa: E402


VOLTAGE_CHANNEL = 0
CURRENT_CHANNEL = 1
TIMESTAMP_CHANNEL = 7


class LiPMDownstreamModel(nn.Module):
    def __init__(self, config, hidden_dim=256, dropout=0.1, normalize_vc=False):
        super().__init__()
        self.patch_len = config.patch_len
        self.normalize_vc = normalize_vc
        self.backbone = BatteryNet(config).backbone
        self.head = nn.Sequential(
            nn.LayerNorm(config.emb_dim),
            nn.Linear(config.emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, samples):
        snippet_count = None
        if samples.dim() == 4:
            batch_size, snippet_count, channel_count, seq_len = samples.shape
            samples = samples.reshape(batch_size * snippet_count, channel_count, seq_len)

        vc, t, t_mask, patch_mask = self._to_lipm_inputs(samples)
        emb = self.backbone(vc, t, t_mask, patch_mask)
        patch_weight = patch_mask.float().unsqueeze(-1)
        pooled = (emb * patch_weight).sum(dim=1) / patch_weight.sum(dim=1).clamp_min(1.0)
        if snippet_count is not None:
            pooled = pooled.view(batch_size, snippet_count, -1).mean(dim=1)
        return self.head(pooled).squeeze(-1)

    def _to_lipm_inputs(self, samples):
        if samples.dim() != 3:
            raise ValueError(f"Expected samples as [B, C, L], got {tuple(samples.shape)}")

        batch_size, channel_count, seq_len = samples.shape
        raw_samples = samples.float()
        samples = raw_samples[:, [VOLTAGE_CHANNEL, CURRENT_CHANNEL], :]
        if self.normalize_vc:
            mean = samples.mean(dim=-1, keepdim=True)
            std = samples.std(dim=-1, keepdim=True)
            samples = (samples - mean) / std.clamp_min(1e-6)
        timestamps = None
        if channel_count > TIMESTAMP_CHANNEL:
            timestamps = raw_samples[:, TIMESTAMP_CHANNEL, :]
        patch_num = int(math.ceil(seq_len / self.patch_len))
        padded_len = patch_num * self.patch_len
        pad_len = padded_len - seq_len
        if pad_len:
            samples = torch.nn.functional.pad(samples, (0, pad_len))
            if timestamps is not None:
                timestamps = torch.nn.functional.pad(timestamps, (0, pad_len))

        vc = samples.transpose(1, 2).contiguous().view(batch_size, patch_num, self.patch_len, 2)
        valid = torch.ones(batch_size, padded_len, dtype=torch.long, device=samples.device)
        if pad_len:
            valid[:, seq_len:] = 0
        valid = valid.view(batch_size, patch_num, self.patch_len)
        patch_mask = (valid.sum(dim=-1) > 0).long()

        if timestamps is None:
            base_t = torch.arange(self.patch_len, dtype=torch.float32, device=samples.device)
            t = base_t.view(1, 1, -1).expand(batch_size, patch_num, -1)
        else:
            t = timestamps.contiguous().view(batch_size, patch_num, self.patch_len)
            first_t = t[:, :, :1]
            t = (t - first_t) * valid.float()
        t = torch.cat([torch.zeros(batch_size, patch_num, 1, device=samples.device), t], dim=-1)
        t_mask = torch.cat([torch.ones(batch_size, patch_num, 1, dtype=torch.long, device=samples.device), valid], dim=-1)
        return vc, t, t_mask, patch_mask


def get_args():
    parser = argparse.ArgumentParser("LiPM downstream fine-tuning on FMAE data")
    parser.add_argument("--pretrained_backbone", default="", help="Path to official LiPM backbone checkpoint.")
    parser.add_argument("--allow_random_init", action="store_true", help="Only for smoke tests; do not report as baseline.")

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=5, type=int)

    parser.add_argument("--fold_num", default=0, type=int)
    parser.add_argument("--brand_num", default=1, type=int)
    parser.add_argument("--downstream", required=True, choices=["capacity", "RUL", "IR"])
    parser.add_argument("--data_percent", default=100, type=int)
    parser.add_argument("--same_normalizer", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--h", default=None, type=int)
    parser.add_argument("--cycle_gap", default=0, type=int)
    parser.add_argument("--num_snippet", default=0, type=int)
    parser.add_argument("--cycle_id", nargs="+", default=None)
    parser.add_argument("--inference_cycle_id", nargs="+", default=None)

    parser.add_argument("--lipm_setting", default=3, type=int, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--patch_len", default=64, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--head_hidden_dim", default=256, type=int)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--normalize_vc", action="store_true")
    parser.add_argument("--no_normalize_vc", action="store_true")
    parser.add_argument("--save_best_ckpt", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lipm_config(args):
    settings = {
        0: (64, 4, 2),
        1: (128, 4, 3),
        2: (256, 8, 6),
        3: (512, 8, 12),
        4: (768, 12, 18),
        5: (1024, 16, 24),
    }
    d_model, n_head, n_layer = settings[args.lipm_setting]
    return SimpleNamespace(
        d_model=d_model,
        emb_dim=d_model,
        n_head=n_head,
        n_layer=n_layer,
        q_k_dim=d_model // n_head,
        v_dim=d_model // n_head,
        dp=args.dropout,
        n_var=2,
        norm="rsm",
        pre_norm=1,
        patch_len=args.patch_len,
        down_dim=256,
        down_n_head=8,
        task="downstream",
    )


def resolve_split_dir(args):
    if args.downstream == "anomaly":
        name = "five_fold_utils_six_brand_all"
    elif args.downstream == "capacity":
        if args.brand_num == 7:
            name = "five_fold_utils_storage_capacity"
        elif args.brand_num in [10, 11, 12, 13]:
            name = "five_fold_utils_lab_capacity"
        elif args.brand_num == 14:
            name = "five_fold_utils_nc_relaxation_capacity"
        elif args.brand_num == -1:
            name = "five_fold_utils_demo"
        else:
            name = "five_fold_utils_EV_capacity"
    elif args.downstream == "IR":
        name = "five_fold_utils_IR"
    elif args.downstream == "RUL":
        name = "five_fold_utils_RUL"
    else:
        raise NotImplementedError(args.downstream)
    return f"demo/{name}" if args.brand_num == -1 else f"five_fold_utils/{name}"


def load_normalizer(args):
    if args.downstream == "anomaly":
        return pickle.load(open("normailze/norm.pkl", "rb"))
    if args.downstream == "capacity":
        if args.brand_num in [7, 10, 11, 12, 13]:
            return None
        return pickle.load(open("normailze/norm.pkl", "rb"))
    if args.downstream in ["IR", "RUL"]:
        return None
    raise NotImplementedError(args.downstream)


def build_loaders(args):
    split_dir = resolve_split_dir(args)
    ind_ood_car_dict = np.load(f"./{split_dir}/ind_odd_dict{args.brand_num}.npz.npy", allow_pickle=True).item()
    all_car_dict = np.load(f"./{split_dir}/all_car_dict.npz.npy", allow_pickle=True).item()

    random.shuffle(ind_ood_car_dict["ind_sorted"])
    random.shuffle(ind_ood_car_dict["ood_sorted"])
    for car_num in ind_ood_car_dict["ind_sorted"] + ind_ood_car_dict["ood_sorted"]:
        random.shuffle(all_car_dict[car_num])

    normalizer = load_normalizer(args)
    train_dataset, normalizer = load_dataset(
        args.fold_num,
        brand_num=args.brand_num,
        same_normalizer=args.same_normalizer,
        car_dict_dir=split_dir,
        downstream=args.downstream,
        data_type="finetune_train",
        normalizer=normalizer,
        dataset_fn=PreprocessNormalizer,
        ind_ood_car_dict=ind_ood_car_dict,
        all_car_dict=all_car_dict,
        num_snippet=args.num_snippet,
        cycle_gap=args.cycle_gap,
        cycle_id=args.cycle_id,
        data_percent=args.data_percent,
        seed=args.seed,
        task="batterybrandmileage",
    )
    test_dataset, _ = load_dataset(
        args.fold_num,
        brand_num=args.brand_num,
        same_normalizer=args.same_normalizer,
        car_dict_dir=split_dir,
        downstream=args.downstream,
        data_type="finetune_test",
        normalizer=normalizer,
        dataset_fn=PreprocessNormalizer,
        ind_ood_car_dict=ind_ood_car_dict,
        all_car_dict=all_car_dict,
        num_snippet=args.num_snippet,
        cycle_gap=args.cycle_gap,
        cycle_id=args.cycle_id,
        data_percent=args.data_percent,
        seed=args.seed,
        task="batterybrandmileage",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=False,
    )
    return train_loader, test_loader


def extract_backbone_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["backbone", "backbone_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a state_dict-like object.")

    prefixes = ["module.backbone.", "backbone.", "module."]
    state = {}
    for key, value in checkpoint.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        if not any(skip in new_key for skip in ["inter_head.", "Q_head.", "head."]):
            state[new_key] = value
    return state


def load_pretrained_backbone(model, path, strict):
    checkpoint = torch.load(path, map_location="cpu")
    state = extract_backbone_state(checkpoint)
    msg = model.backbone.load_state_dict(state, strict=strict)
    print(f"Loaded LiPM backbone from {path}")
    print(msg)


def move_batch(batch, device):
    moved = []
    for item in batch:
        moved.append(item.to(device, non_blocking=True) if isinstance(item, torch.Tensor) else item)
    return moved


def train_one_epoch(model, loader, criterion, optimizer, device, args, epoch):
    model.train()
    total_loss = 0.0
    total_count = 0
    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        samples, targets = batch[0], batch[1].float()
        logits = model(samples)
        loss = criterion(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * samples.shape[0]
        total_count += samples.shape[0]
        if step % 200 == 0:
            print(f"epoch {epoch} step {step}/{len(loader)} loss {loss.item():.6f}")
    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, args):
    model.eval()
    rows = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        samples, targets, cars = batch[0], batch[1].float(), batch[2]
        logits = model(samples)
        loss = criterion(logits, targets)
        total_loss += loss.item() * samples.shape[0]
        total_count += samples.shape[0]

        logits_np = logits.detach().cpu().numpy()
        labels_np = targets.detach().cpu().numpy()
        cars_np = cars.detach().cpu().numpy() if isinstance(cars, torch.Tensor) else np.asarray(cars)
        if args.downstream == "RUL":
            current_cycle = batch[3].detach().cpu().numpy()
            rul_cycle = batch[4].detach().cpu().numpy()
            rows.append(np.stack([labels_np, cars_np, logits_np, current_cycle, rul_cycle], axis=1))
        else:
            rows.append(np.stack([labels_np, cars_np, logits_np], axis=1))

    result = np.concatenate(rows, axis=0) if rows else np.empty((0, 3))
    mean_loss = total_loss / max(total_count, 1)
    if args.downstream == "anomaly":
        metric = util.evaluation.evaluation(result, brand_num=args.brand_num, h=args.h)
        return result, mean_loss, {"auroc": metric}
    if args.downstream == "RUL":
        rmse, pct = evaluate_RUL(result, args.inference_cycle_id)
        return result, mean_loss, {"rmse": math.sqrt(mean_loss), "cell_rmse": rmse, "cell_percentage_error": pct}
    return result, mean_loss, {"rmse": math.sqrt(mean_loss)}


def main():
    args = get_args()
    if not args.pretrained_backbone and not args.allow_random_init:
        raise SystemExit(
            "No LiPM pretrained backbone was provided. The official public release currently "
            "does not provide weights, so formal baseline fine-tuning is blocked. "
            "Use --allow_random_init only for smoke tests."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")

    set_seed(args.seed)
    device = torch.device(args.device)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = make_lipm_config(args)
    model = LiPMDownstreamModel(
        config,
        hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        normalize_vc=args.normalize_vc and not args.no_normalize_vc,
    )
    if args.pretrained_backbone:
        load_pretrained_backbone(model, args.pretrained_backbone, args.strict_load)
    else:
        print("WARNING: running with random LiPM initialization. This is only valid as a smoke test.")

    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    model.to(device)
    train_loader, test_loader = build_loaders(args)

    if args.downstream == "anomaly":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(args)
    print(f"train samples: {len(train_loader.dataset)}, test samples: {len(test_loader.dataset)}")
    start_time = time.time()
    best_metric = None
    best_result = None

    for epoch in range(args.epochs):
        if args.downstream == "RUL" and args.cycle_gap > 0 and hasattr(train_loader.dataset, "shuffle_finetune"):
            train_loader.dataset.shuffle_finetune()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, args, epoch)
        result, test_loss, metrics = evaluate(model, test_loader, criterion, device, args)
        print(f"epoch {epoch} train_loss {train_loss:.9f} test_loss {test_loss:.9f} metrics {metrics}")

        if args.downstream == "anomaly":
            current = metrics["auroc"]
        elif args.downstream == "RUL":
            current = -metrics["cell_rmse"]
        else:
            current = -metrics["rmse"]
        if best_metric is None or current > best_metric:
            best_metric = current
            best_result = result
            if args.output_dir:
                np.save(os.path.join(args.output_dir, "res.npy"), result)
                if args.save_best_ckpt:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_metric": best_metric,
                            "metrics": metrics,
                            "args": vars(args),
                        },
                        os.path.join(args.output_dir, "checkpoint-best.pth"),
                    )

    if args.epochs == 0:
        best_result, test_loss, metrics = evaluate(model, test_loader, criterion, device, args)
        print(f"eval_only test_loss {test_loss:.9f} metrics {metrics}")
        if args.output_dir:
            np.save(os.path.join(args.output_dir, "res.npy"), best_result)

    total_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time}")


if __name__ == "__main__":
    main()
