"""Upscale an arbitrary LR image x4 with trained checkpoints and compare against
other upscaled versions of the same image (no ground truth needed).

Reports LR-consistency PSNR: downscale each x4 output back with MATLAB bicubic and
compare to the input. High = output is faithful to the input; it is NOT a quality
score (bicubic upsampling scores well while being blurry).

  python tools/upscale_pair.py --lr in.jpg --ref "plain resize=a.jpg" --ref "AI upscaler=b.jpg"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from csea.imresize import imresize  # noqa: E402
from test import build  # noqa: E402


def to_t(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255)


def to_u8(t):
    return t.clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).numpy()


def lr_consistency(out_u8, lr_u8):
    down = to_u8(imresize(to_t(out_u8), 0.25))
    mse = np.mean((down.astype(np.float64) - lr_u8.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", required=True)
    ap.add_argument("--ref", action="append", default=[], help="label=path of another x4 result")
    ap.add_argument("--ckpt", action="append", default=[], help="label=path to checkpoint")
    ap.add_argument("--out", default="experiments/demo")
    ap.add_argument("--crop", type=int, nargs=3, default=[150, 170, 160], help="x y size in HR pixels")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lr = np.array(Image.open(args.lr).convert("RGB"))
    h, w = lr.shape[:2]
    panels = [("Input (nearest x4)", np.array(Image.fromarray(lr).resize((w * 4, h * 4), Image.NEAREST))),
              ("Bicubic", to_u8(imresize(to_t(lr), 4)))]
    for spec in args.ref:
        label, p = spec.split("=", 1)
        im = np.array(Image.open(p).convert("RGB"))
        assert im.shape[:2] == (h * 4, w * 4), f"{p} is {im.shape[:2]}, expected {(h * 4, w * 4)}"
        panels.append((label, im))
    for spec in args.ckpt:
        label, p = spec.split("=", 1)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        model = build(ck).eval()  # CPU: leaves the GPU to the running training job
        with torch.no_grad():
            sr = to_u8(model(to_t(lr)))
        Image.fromarray(sr).save(out / f"{Path(args.lr).stem}_{label.replace(' ', '_')}.png")
        panels.append((label, sr))

    print(f"{'output':28s} LR-consistency PSNR (dB)")
    for label, im in panels[1:]:
        print(f"{label:28s} {lr_consistency(im, lr):6.2f}")

    # comparison sheet: full images (row 1) + zoomed crop (row 2)
    x, y, c = args.crop
    tile, gap, top = 256, 6, 22
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (len(panels) * (tile + gap) + gap, 2 * (tile + top) + gap), "white")
    d = ImageDraw.Draw(sheet)
    for i, (label, im) in enumerate(panels):
        px = gap + i * (tile + gap)
        full = Image.fromarray(im).resize((tile, tile), Image.LANCZOS)
        sheet.paste(full, (px, top))
        s = tile / im.shape[1]
        d.rectangle([px + x * s, top + y * s, px + (x + c) * s, top + (y + c) * s], outline="red", width=2)
        crop = Image.fromarray(im[y:y + c, x:x + c]).resize((tile, tile), Image.NEAREST)
        sheet.paste(crop, (px, 2 * top + tile + gap))
        d.text((px, 5), label, fill="black", font=font)
        d.text((px, top + tile + gap + 5), "zoom", fill="black", font=font)
    p = out / f"{Path(args.lr).stem}_comparison.png"
    sheet.save(p)
    print("wrote", p)


if __name__ == "__main__":
    main()
