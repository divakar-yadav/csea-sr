"""Unpack DIV2K + benchmarks and build x{scale} LR/HR pairs (MATLAB bicubic protocol).

Output layout (data/processed/):
  DIV2K_train/{HR,LRx4}/0001.npy     uint8 HxWx3, memory-mappable for fast random crops
  bench/<Set>/{HR,LRx4}/<name>.png   HR mod-cropped to a multiple of the scale
"""
import argparse
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from csea.imresize import bicubic_downsample_uint8

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def modcrop(img, s):
    h, w = img.shape[:2]
    return img[: h - h % s, : w - w % s]


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()
    raw, out, s = Path(args.raw), Path(args.out), args.scale
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ext = raw / "extracted"

    # ---- DIV2K -------------------------------------------------------------
    for split in ("train", "valid"):
        z = raw / f"DIV2K_{split}_HR.zip"
        tgt = ext / f"DIV2K_{split}_HR"
        if not tgt.exists():
            print(f"unzipping {z.name} ...")
            with zipfile.ZipFile(z) as f:
                f.extractall(ext)
        files = sorted(p for p in tgt.iterdir() if p.suffix.lower() in IMG_EXT)
        hr_dir, lr_dir = out / f"DIV2K_{split}" / "HR", out / f"DIV2K_{split}" / f"LRx{s}"
        hr_dir.mkdir(parents=True, exist_ok=True); lr_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(files):
            if (lr_dir / f"{p.stem}.npy").exists():
                continue
            hr = modcrop(load_rgb(p), s)
            np.save(hr_dir / f"{p.stem}.npy", hr)
            np.save(lr_dir / f"{p.stem}.npy", bicubic_downsample_uint8(hr, s, dev))
            if i % 100 == 0:
                print(f"  DIV2K_{split} {i}/{len(files)}")
        print(f"DIV2K_{split}: {len(files)} images")

    # ---- benchmarks --------------------------------------------------------
    for tgz in sorted((raw / "bench").glob("*_HR.tar.gz")):
        name = tgz.name.replace("_HR.tar.gz", "")
        d = ext / "bench" / name
        if not d.exists():
            d.mkdir(parents=True)
            with tarfile.open(tgz) as t:
                t.extractall(d, filter="data")
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
        hr_dir, lr_dir = out / "bench" / name / "HR", out / "bench" / name / f"LRx{s}"
        hr_dir.mkdir(parents=True, exist_ok=True); lr_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            hr = modcrop(load_rgb(p), s)
            Image.fromarray(hr).save(hr_dir / f"{p.stem}.png")
            Image.fromarray(bicubic_downsample_uint8(hr, s, dev)).save(lr_dir / f"{p.stem}.png")
        print(f"{name}: {len(files)} images")


if __name__ == "__main__":
    main()
