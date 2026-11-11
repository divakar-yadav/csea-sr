# CSEAT: Region-Restricted Long-Range Attention for Lightweight Super-Resolution

Code and weights for a lightweight (~0.9M parameter) ×4 image super-resolution transformer that
combines **local window attention** with **region-restricted long-range attention**, including
*Cross-Scale Exemplar Attention* (CSEA): keys come from an f×-pooled copy of the feature map, and
each coarse match returns its full-resolution f×f **child block** as the value.

## Results (×4, PSNR/SSIM on the Y channel, 4-px border shave)

Equal-budget ablation: 50K iterations, batch 16, 64×64 LR patches, DIV2K, single 8 GB GPU.

| Long-range branch | Params | MACs (720p) | Set5 | Set14 | BSD100 | Urban100 |
|---|---|---|---|---|---|---|
| none (window only) | 785K | 52.3G | 32.23/0.8950 | 28.62/0.7826 | 27.61/0.7378 | 26.19/0.7889 |
| pooled values (SRA) | 919K | 75.7G | 32.31/0.8957 | 28.65/0.7833 | 27.62/0.7384 | 26.29/0.7919 |
| CSEA (child-block values) | 894K | 75.7G | 32.23/0.8950 | 28.63/0.7829 | 27.62/0.7384 | 26.30/0.7922 |
| pooled, test-time region 64 | 919K | 124.0G | 32.32/0.8957 | 28.66/0.7832 | 27.62/0.7381 | 26.33/0.7927 |
| CSEA, test-time region 64 | 894K | 124.0G | 32.24/0.8951 | 28.64/0.7826 | 27.61/0.7381 | 26.34/0.7929 |

Takeaways: region-restricted long-range attention adds ~+0.1 dB on Urban100 (repetitive structure),
enlarging the region at test time adds a further +0.04 dB without retraining, and child-block values
perform on par with pooled values at equal MACs with fewer parameters. A longer-trained main model
will be added to `weights/`.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build for your GPU
pip install -r requirements.txt
```

## Data

Download into `data/raw/`:
- `DIV2K_train_HR.zip`, `DIV2K_valid_HR.zip` from the [DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- benchmark HR archives as `data/raw/bench/<Set>_HR.tar.gz` (Set5, Set14, BSD100, Urban100)

Then build the LR/HR pairs with MATLAB-compatible bicubic downsampling:

```bash
python prepare_data.py            # writes data/processed/
```

## Evaluate the released weights

```bash
python test.py --ckpt weights/abl50k_xscale.pt                 # PSNR/SSIM, params, MACs, runtime
python test.py --ckpt weights/abl50k_xscale.pt --region 64     # larger search region at test time
python test.py --ckpt weights/abl50k_xscale.pt --save-images   # also write SR images
```

Upscale your own image (no ground truth needed):

```bash
python tools/upscale_pair.py --lr my_image.png --ckpt "CSEAT=weights/abl50k_xscale.pt" --out demo
```

Note: the models are trained on clean bicubic degradations; heavily JPEG-compressed inputs will
have their block artefacts sharpened.

## Train

```bash
python train.py --exp my_run --global-mode xscale --iters 50000 --warmup 1000   # CSEA
python train.py --exp my_run --global-mode pool   --iters 50000 --warmup 1000   # pooled-value baseline
python train.py --exp my_run --global-mode none   --iters 50000 --warmup 1000   # window attention only
```

Training is resumable (`experiments/<exp>/last.pt`) and logs to Weights & Biases
(`--wandb online|offline|disabled`). Batch 16 uses about 4–5 GB of GPU memory with bfloat16.

## Repository layout

```
csea/model.py        CSEAT network, window attention, region-restricted CSEA / pooled attention
csea/imresize.py     MATLAB-compatible bicubic resize
csea/metrics.py      Y-channel PSNR / SSIM
prepare_data.py      dataset preparation
train.py, test.py    training and benchmarking
tools/               bicubic protocol check, complexity measurement, demo upscaling
weights/             released checkpoints (EMA weights)
```
