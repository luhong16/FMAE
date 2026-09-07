#!/usr/bin/env python3
"""Pretrain LiPM on this repository's EV snippets."""

import argparse
import datetime
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
LIPM_ROOT = THIS_DIR.parent


def find_repo_root(start):
    for path in [start] + list(start.parents):
        if (path / "five_fold_utils").exists() and (path / "util" / "datasets.py").exists():
            return path
    raise RuntimeError(f"Could not find FMAE repository root from {start}")


REPO_ROOT = find_repo_root(THIS_DIR)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(LIPM_ROOT))

from LiPM.battery_model import BatteryNet  # noqa: E402


VOLTAGE_CHANNEL = 0
CURRENT_CHANNEL = 1
TIMESTAMP_CHANNEL = 7


class EVSnippetPretrainDataset(Dataset):
    def __init__(self, paths, patch_len, patch_num, normalize_vc=True, max_samples=None):
        self.paths = list(paths)
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        self.patch_len = patch_len
        self.patch_num = patch_num
        self.seq_len = patch_len * patch_num
        self.normalize_vc = normalize_vc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        sample, _metadata = torch.load(self.paths[index], map_location="cpu")
        sample = np.asarray(sample, dtype=np.float32)
        if sample.ndim != 2 or sample.shape[1] <= TIMESTAMP_CHANNEL:
            raise ValueError(f"Unexpected sample shape in {self.paths[index]}: {sample.shape}")

        sample = self._fit_length(sample)
        volt = sample[:, VOLTAGE_CHANNEL]
        current = sample[:, CURRENT_CHANNEL]
        timestamp = sample[:, TIMESTAMP_CHANNEL]
        timestamp = self._relative_monotonic_time(timestamp)
        qd_qc = self._integrate_qd_qc(current, timestamp)

        vc = np.stack([volt, current], axis=-1)
        if self.normalize_vc:
            mean = vc.mean(axis=0, keepdims=True)
            std = vc.std(axis=0, keepdims=True)
            vc = (vc - mean) / np.maximum(std, 1e-6)

        vc = vc.reshape(self.patch_num, self.patch_len, 2)
        qd_qc = qd_qc.reshape(self.patch_num, self.patch_len, 2)
        t = timestamp.reshape(self.patch_num, self.patch_len)
        t = t - t[:, :1]

        t_mask = np.ones((self.patch_num, self.patch_len + 1), dtype=np.int64)
        patch_mask = np.ones((self.patch_num,), dtype=np.int64)
        t = np.concatenate([np.zeros((self.patch_num, 1), dtype=np.float32), t], axis=1)

        return (
            torch.from_numpy(vc.astype(np.float32)),
            torch.from_numpy(qd_qc.astype(np.float32)),
            torch.from_numpy(t.astype(np.float32)),
            torch.from_numpy(t_mask),
            torch.from_numpy(patch_mask),
        )

    def _fit_length(self, sample):
        if len(sample) == self.seq_len:
            return sample
        if len(sample) > self.seq_len:
            return sample[:self.seq_len]
        pad = np.zeros((self.seq_len - len(sample), sample.shape[1]), dtype=np.float32)
        if len(sample) > 0:
            pad[:] = sample[-1:]
        return np.concatenate([sample, pad], axis=0)

    @staticmethod
    def _relative_monotonic_time(timestamp):
        timestamp = np.asarray(timestamp, dtype=np.float32)
        timestamp = timestamp - timestamp[0]
        timestamp = np.maximum.accumulate(timestamp)
        return timestamp

    @staticmethod
    def _integrate_qd_qc(current, timestamp):
        current = np.asarray(current, dtype=np.float32)
        timestamp = np.asarray(timestamp, dtype=np.float32)
        dt = np.diff(timestamp, prepend=timestamp[0])
        dt = np.maximum(dt, 0.0)

        discharge_current = np.maximum(current, 0.0)
        charge_current = np.maximum(-current, 0.0)
        discharge_avg = 0.5 * (discharge_current + np.roll(discharge_current, 1))
        charge_avg = 0.5 * (charge_current + np.roll(charge_current, 1))
        discharge_avg[0] = 0.0
        charge_avg[0] = 0.0
        qd = np.cumsum(discharge_avg * dt) / 3600.0
        qc = np.cumsum(charge_avg * dt) / 3600.0
        qd_qc = np.stack([qd, qc], axis=-1)

        scale = np.maximum(np.max(np.abs(qd_qc), axis=0, keepdims=True), 1e-6)
        return qd_qc / scale


def get_args():
    parser = argparse.ArgumentParser("LiPM EV pretraining on FMAE snippets")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "experiment" / "lipm" / "pretrained_model"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=5, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--epochs", default=18, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--max_samples", default=None, type=int)

    parser.add_argument("--lipm_setting", default=3, type=int, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--patch_len", default=64, type=int)
    parser.add_argument("--patch_num", default=2, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--patch_ratio", default=0.3, type=float)
    parser.add_argument("--channel_ratio", default=0.3, type=float)
    parser.add_argument("--weight_MAE", default=1.0, type=float)
    parser.add_argument("--weight_Q", default=1.0, type=float)
    parser.add_argument("--no_normalize_vc", action="store_true")
    parser.add_argument("--local_rank", default=-1, type=int)
    return parser.parse_args()


def make_config(args):
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
        patch_num=args.patch_num,
        down_dim=256,
        down_n_head=8,
        task="ir_pretrain",
        max_iter=max(args.epochs, 1),
        patch_ratio=args.patch_ratio,
        channel_ratio=args.channel_ratio,
        weight_MAE=args.weight_MAE,
        weight_Q=args.weight_Q,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed(args):
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        args.distributed = True
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        if args.local_rank < 0:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()
    return args


def is_main_process(args):
    return args.rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def collect_ev_paths():
    split_dir = REPO_ROOT / "five_fold_utils" / "five_fold_utils_six_brand_all"
    all_car_dict = np.load(split_dir / "all_car_dict.npz.npy", allow_pickle=True).item()
    paths = []
    for car in sorted(all_car_dict):
        for entry in all_car_dict[car]:
            path = entry[0] if isinstance(entry, (list, tuple)) else entry
            if "data/EV/data/" in path:
                paths.append(str(REPO_ROOT / path))
    return paths


def save_checkpoint(model, optimizer, args, epoch, output_dir):
    raw_model = unwrap_model(model)
    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "backbone": raw_model.backbone.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(payload, output_dir / f"checkpoint-{epoch}.pth")
    torch.save(raw_model.backbone.state_dict(), output_dir / f"backbone-{epoch}.pth")
    torch.save(payload, output_dir / "checkpoint-last.pth")
    torch.save(raw_model.backbone.state_dict(), output_dir / "backbone-last.pth")


def move_batch(batch, device):
    return [item.to(device, non_blocking=True) for item in batch]


def main():
    args = get_args()
    args = init_distributed(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if is_main_process(args):
        output_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed:
        dist.barrier()

    paths = collect_ev_paths()
    random.shuffle(paths)
    if not paths:
        raise SystemExit("No EV snippets found from five_fold_utils_six_brand_all.")
    dataset = EVSnippetPretrainDataset(
        paths,
        patch_len=args.patch_len,
        patch_num=args.patch_num,
        normalize_vc=not args.no_normalize_vc,
        max_samples=args.max_samples,
    )
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True) if args.distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )

    config = make_config(args)
    if args.distributed:
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device(args.device)
    model = BatteryNet(config).to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    if is_main_process(args):
        print(args)
        print(f"EV pretraining snippets: {len(dataset)}")
        print(f"LiPM config: {config}")
    start_time = time.time()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        count = 0
        for step, batch in enumerate(loader):
            batch = move_batch(batch, device)
            loss, loss_dict = model(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = batch[0].shape[0]
            epoch_loss += loss.item() * batch_size
            count += batch_size
            if is_main_process(args) and step % 100 == 0:
                details = " ".join(f"{k}={v:.6f}" for k, v in sorted(loss_dict.items()))
                print(f"epoch {epoch} step {step}/{len(loader)} {details}")
        if args.distributed:
            stats = torch.tensor([epoch_loss, count], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            epoch_loss, count = stats.tolist()
        mean_loss = epoch_loss / max(count, 1)
        if is_main_process(args):
            print(f"epoch {epoch} mean_loss {mean_loss:.9f}")
        if is_main_process(args) and ((epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs):
            save_checkpoint(model, optimizer, args, epoch, output_dir)

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    if is_main_process(args):
        print(f"Pretraining time {elapsed}")
        print(f"Use this backbone for downstream fine-tuning: {output_dir / 'backbone-last.pth'}")
    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
