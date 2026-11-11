"""Train CSEAT (x4 lightweight SR) on DIV2K. Resumable; logs to Weights & Biases.

Examples:
  python train.py --exp csea_main --global-mode xscale --iters 200000
  python train.py --exp abl_none  --global-mode none   --iters 50000
  python train.py --exp smoke --iters 200 --val-every 100 --wandb disabled
"""
import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

from csea.model import CSEAT, count_params
from csea.metrics import psnr_y, ssim_y

GB = 2 ** 30


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class DIV2KPatches(Dataset):
    def __init__(self, root, scale, patch, length):
        root = Path(root)
        self.hr = sorted((root / "HR").glob("*.npy"))
        self.lr = [root / f"LRx{scale}" / p.name for p in self.hr]
        assert self.hr, f"no data in {root} - run prepare_data.py"
        self.s, self.p, self.length = scale, patch, length
        self._cache = {}

    def __len__(self):
        return self.length

    def _arr(self, path):
        if path not in self._cache:
            self._cache[path] = np.load(path, mmap_mode="r")
        return self._cache[path]

    def __getitem__(self, i):
        j = random.randrange(len(self.hr))
        lr, hr = self._arr(self.lr[j]), self._arr(self.hr[j])
        p, s = self.p, self.s
        y, x = random.randrange(lr.shape[0] - p + 1), random.randrange(lr.shape[1] - p + 1)
        lr = np.array(lr[y:y + p, x:x + p])
        hr = np.array(hr[y * s:(y + p) * s, x * s:(x + p) * s])
        if random.random() < 0.5:
            lr, hr = lr[:, ::-1], hr[:, ::-1]
        if random.random() < 0.5:
            lr, hr = lr[::-1], hr[::-1]
        if random.random() < 0.5:
            lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return to_t(lr), to_t(hr)              # uint8; normalised on GPU


def worker_init(wid):
    s = torch.initial_seed() % 2 ** 32
    random.seed(s); np.random.seed(s)


def load_val(root, scale, n):
    root = Path(root)
    hrs = sorted((root / "HR").glob("*.npy"))[:n]
    return [(p.stem, np.load(root / f"LRx{scale}" / p.name), np.load(p)) for p in hrs]


@torch.no_grad()
def evaluate(model, items, scale, device):
    model.eval()
    ps, ss = [], []
    for _, lr, hr in items:
        x = torch.from_numpy(lr).permute(2, 0, 1)[None].float().div(255).to(device)
        sr = model(x).clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
        ps.append(psnr_y(sr, hr, scale)); ss.append(ssim_y(sr, hr, scale))
    return float(np.mean(ps)), float(np.mean(ss))


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay):
        self.decay, self.model = decay, copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.lerp_(m, 1 - self.decay)


def lr_at(it, base, warmup, total, min_lr):
    if it < warmup:
        return base * (it + 1) / warmup
    t = (it - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * min(1.0, t)))


def save_atomic(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--global-mode", default="xscale", choices=["none", "pool", "xscale"])
    ap.add_argument("--factor", type=int, default=2)
    ap.add_argument("--region", type=int, default=32, help="long-range attention region (0=global)")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64, help="LR patch size")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-ckpt", action="store_true", help="activation checkpointing (saves memory)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--val-every", type=int, default=5000)
    ap.add_argument("--val-images", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--wandb", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb-project", default="csea-sr")
    return ap.parse_args()


def main():
    args = parse()
    out = Path(args.out) / args.exp
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    print(f"[gpu] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    model = CSEAT(scale=args.scale, dim=args.dim, groups=args.groups, depth=args.depth,
                  global_mode=args.global_mode, factor=args.factor, region=args.region).to(dev)
    n_params = count_params(model)
    print(f"[model] {args.global_mode} | params {n_params / 1e3:.1f}K")
    ema = EMA(model, args.ema)               # copy BEFORE patching forward for checkpointing
    if args.grad_ckpt:
        for g in model.body:
            g.forward = (lambda f: (lambda x: checkpoint(f, x, use_reentrant=False)))(g.forward)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    start, wandb_id, best = 0, None, -1.0

    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); ema.model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start, wandb_id, best = ck["iter"], ck.get("wandb_id"), ck.get("best", -1.0)
        random.setstate(ck["py_rng"]); np.random.set_state(ck["np_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"[resume] from iter {start}")

    run = None
    if args.wandb != "disabled":
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.exp, id=wandb_id, resume="allow",
                         mode=args.wandb, dir=str(out),
                         config={**vars(args), "params": n_params})
        wandb_id = run.id
        print(f"[wandb] {run.url}")
    with open(out / "config.json", "w") as f:
        json.dump({**vars(args), "params": n_params}, f, indent=2)

    ds = DIV2KPatches(Path(args.data) / "DIV2K_train", args.scale, args.patch,
                      length=(args.iters - start) * args.batch_size + args.batch_size)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                    drop_last=True, persistent_workers=args.workers > 0, worker_init_fn=worker_init,
                    prefetch_factor=4 if args.workers > 0 else None)
    val = load_val(Path(args.data) / "DIV2K_valid", args.scale, args.val_images)

    def ckpt(it):
        return {"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                "iter": it, "args": vars(args), "params": n_params, "wandb_id": wandb_id,
                "best": best, "py_rng": random.getstate(), "np_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state()}

    model.train()
    it, t0, loss_acc, n_acc = start, time.time(), 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    for lr_u8, hr_u8 in dl:
        if it >= args.iters:
            break
        lr_img = lr_u8.to(dev, non_blocking=True).float().div_(255)
        hr_img = hr_u8.to(dev, non_blocking=True).float().div_(255)
        cur_lr = lr_at(it, args.lr, args.warmup, args.iters, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sr = model(lr_img)
        loss = F.l1_loss(sr.float(), hr_img)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        ema.update(model)
        it += 1
        loss_acc += loss.item(); n_acc += 1

        if it % args.log_every == 0:
            dt = time.time() - t0
            ips = n_acc / dt
            eta_h = (args.iters - it) / ips / 3600
            mem = torch.cuda.max_memory_allocated() / GB
            print(f"it {it:>7}/{args.iters} | loss {loss_acc / n_acc:.4f} | lr {cur_lr:.2e} | "
                  f"{ips:.2f} it/s | mem {mem:.2f} GB | ETA {eta_h:.1f} h", flush=True)
            if run:
                run.log({"train/l1": loss_acc / n_acc, "train/lr": cur_lr, "train/grad_norm": float(gnorm),
                         "perf/it_per_s": ips, "perf/patches_per_s": ips * args.batch_size,
                         "perf/peak_mem_gb": mem}, step=it)
            t0, loss_acc, n_acc = time.time(), 0.0, 0

        if it % args.val_every == 0 or it == args.iters:
            p, s = evaluate(ema.model, val, args.scale, dev)
            print(f"[val] it {it} | DIV2K-val{len(val)} PSNR {p:.3f} SSIM {s:.4f} (EMA)", flush=True)
            if run:
                run.log({"val/psnr_y": p, "val/ssim_y": s}, step=it)
            if p > best:
                best = p
                save_atomic({"ema": ema.model.state_dict(), "iter": it, "val_psnr": p,
                             "args": vars(args), "params": n_params}, out / "best.pt")
            model.train()
            t0 = time.time()

        if it % args.ckpt_every == 0 or it == args.iters:
            save_atomic(ckpt(it), last)

    save_atomic({"ema": ema.model.state_dict(), "iter": it, "args": vars(args),
                 "params": n_params}, out / "final.pt")
    if run:
        run.summary["best_val_psnr"] = best
        run.finish()
    print(f"done: {out.resolve()}")


if __name__ == "__main__":
    main()
