"""Benchmark a trained CSEAT checkpoint: PSNR/SSIM (Y, border=scale) on
Set5/Set14/BSD100/Urban100, params, FLOPs (1280x720 output) and GPU runtime.

  python test.py --ckpt experiments/csea_main/final.pt [--save-images]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from csea.metrics import psnr_y, ssim_y
from csea.model import CSEAT, count_params

SETS = ("Set5", "Set14", "BSD100", "Urban100")


def build(ck, region=None):
    a = ck["args"]
    m = CSEAT(scale=a["scale"], dim=a["dim"], groups=a["groups"], depth=a["depth"],
              global_mode=a["global_mode"], factor=a["factor"],
              region=a.get("region", 32) if region is None else region)
    m.load_state_dict(ck["ema"] if "ema" in ck else ck["model"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/processed/bench")
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--save-images", action="store_true")
    ap.add_argument("--self-ensemble", action="store_true", help="x8 geometric self-ensemble (+)")
    ap.add_argument("--region", type=int, default=None, help="override region size at test time")
    args = ap.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build(ck, args.region).to(dev).eval()
    s = ck["args"]["scale"]
    tag = ("results_plus" if args.self_ensemble else "results") + \
          (f"_region{args.region}" if args.region is not None else "")
    out_dir = Path(args.ckpt).parent / tag
    out_dir.mkdir(exist_ok=True)

    def run(x):
        if not args.self_ensemble:
            return model(x)
        outs = []
        for t in range(8):
            y = x.flip(-1) if t & 1 else x
            y = y.flip(-2) if t & 2 else y
            y = y.transpose(-1, -2) if t & 4 else y
            o = model(y)
            o = o.transpose(-1, -2) if t & 4 else o
            o = o.flip(-2) if t & 2 else o
            o = o.flip(-1) if t & 1 else o
            outs.append(o)
        return torch.stack(outs).mean(0)

    res = {"ckpt": str(args.ckpt), "iter": ck.get("iter"), "params": count_params(model),
           "global_mode": ck["args"]["global_mode"]}
    with torch.no_grad():
        # FLOPs for a 1280x720 output (standard for lightweight SR)
        x = torch.rand(1, 3, 720 // s, 1280 // s, device=dev)
        fc = FlopCounterMode(display=False)
        with fc:
            model(x)
        res["gflops_720p"] = fc.get_total_flops() / 2 / 1e9  # report multiply-adds (MACs) as "FLOPs" like SR papers
        for _ in range(3):
            model(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(10):
            model(x)
        torch.cuda.synchronize()
        res["ms_720p"] = (time.time() - t) / 10 * 1000

        for name in args.sets:
            hr_dir, lr_dir = Path(args.data) / name / "HR", Path(args.data) / name / f"LRx{s}"
            ps, ss = [], []
            for hp in sorted(hr_dir.glob("*.png")):
                hr = np.array(Image.open(hp).convert("RGB"))
                lr = np.array(Image.open(lr_dir / hp.name).convert("RGB"))
                xin = torch.from_numpy(lr).permute(2, 0, 1)[None].float().div(255).to(dev)
                sr = run(xin).clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()
                ps.append(psnr_y(sr, hr, s)); ss.append(ssim_y(sr, hr, s))
                if args.save_images:
                    (out_dir / name).mkdir(exist_ok=True)
                    Image.fromarray(sr).save(out_dir / name / hp.name)
            res[name] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(ps)}
            print(f"{name:9s} PSNR {res[name]['psnr']:.2f}  SSIM {res[name]['ssim']:.4f}  ({len(ps)} imgs)")
    print(f"params {res['params'] / 1e3:.0f}K | {res['gflops_720p']:.1f}G MACs @720p | "
          f"{res['ms_720p']:.0f} ms @720p")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
