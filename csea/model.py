"""CSEAT: lightweight SR transformer with local window attention + global
Cross-Scale Exemplar Attention (CSEA).

Global branch variants (`global_mode`), used for the ablation:
  none   - local window attention only
  pool   - plain long-range attention: queries at LR resolution, keys AND values from the
           f x f average-pooled feature map (spatial-reduction attention)
Both long-range variants attend within R x R regions (default 32, shifted by R/2 in
alternate layers); region=0 gives fully global attention.
  xscale - CSEA (ours): keys from the pooled map (a "coarse" copy of the image), but each
           value is the full-resolution f x f *child block* of the matched coarse location
           (pixel-unshuffled), i.e. a higher-resolution exemplar of the query's patch,
           exploiting cross-scale patch recurrence. Parameter count <= `pool`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Local: (shifted) window self-attention
# --------------------------------------------------------------------------- #
class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.h, self.ws, self.shift = heads, window, shift
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.rel_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window),
                                            indexing="ij")).flatten(1)          # 2,N
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + window - 1
        self.register_buffer("rel_index", rel[..., 0] * (2 * window - 1) + rel[..., 1],
                             persistent=False)
        self._mask_cache = {}

    def _shift_mask(self, H, W, device):
        key = (H, W, str(device))
        if key not in self._mask_cache:
            ws, s = self.ws, self.ws // 2
            img = torch.zeros(H, W, device=device)
            cnt = 0
            for hs in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                for wsl in (slice(0, -ws), slice(-ws, -s), slice(-s, None)):
                    img[hs, wsl] = cnt
                    cnt += 1
            win = img.view(H // ws, ws, W // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            m = win[:, None, :] != win[:, :, None]                               # nW,N,N
            self._mask_cache[key] = torch.zeros(m.shape, device=device).masked_fill(m, -100.0)
        return self._mask_cache[key]

    def forward(self, x):                      # x: B,H,W,C  (H,W multiples of window)
        B, H, W, C = x.shape
        ws, h = self.ws, self.h
        if self.shift:
            x = torch.roll(x, (-(ws // 2), -(ws // 2)), dims=(1, 2))
        nh, nw = H // ws, W // ws
        t = x.view(B, nh, ws, nw, ws, C).permute(0, 1, 3, 2, 4, 5).reshape(B, nh * nw, ws * ws, C)
        nW, N = nh * nw, ws * ws
        qkv = self.qkv(t).view(B * nW, N, 3, h, C // h).permute(2, 0, 3, 1, 4)    # 3,B*nW,h,N,d
        q, k, v = qkv[0].contiguous(), qkv[1].contiguous(), qkv[2].contiguous()
        bias = self.rel_bias[self.rel_index].permute(2, 0, 1)[None]              # 1,h,N,N
        if self.shift:
            bias = bias + self._shift_mask(H, W, x.device)[:, None]              # nW,h,N,N
        bias = bias.to(q.dtype).expand(nW, h, N, N)[None].expand(B, nW, h, N, N)
        bias = bias.reshape(B * nW, h, N, N)                                     # 4-D for fused kernel
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)              # B*nW,h,N,d
        o = o.view(B, nW, h, N, C // h).permute(0, 1, 3, 2, 4).reshape(B, nh, nw, ws, ws, C)
        o = o.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        if self.shift:
            o = torch.roll(o, (ws // 2, ws // 2), dims=(1, 2))
        return self.proj(o)


# --------------------------------------------------------------------------- #
# Global: pooled attention vs. Cross-Scale Exemplar Attention
# --------------------------------------------------------------------------- #
class GlobalAttention(nn.Module):
    """Long-range attention inside R x R regions (R >> local window), optionally shifted
    by R/2. Cross-scale recurrence is strongest in a patch's neighbourhood (Zontak &
    Irani, CVPR 2011), so regions keep cost linear in image size. region=0 -> whole image."""

    def __init__(self, dim, heads, factor=2, mode="xscale", region=32, shift=False, q_chunk=8192):
        super().__init__()
        assert mode in ("pool", "xscale")
        assert region == 0 or region % factor == 0
        self.region, self.shift = region, shift
        self.h, self.f, self.mode, self.q_chunk = heads, factor, mode, q_chunk
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        if mode == "pool":
            self.v = nn.Linear(dim, dim)                 # on pooled features
        else:
            assert dim % (factor ** 2) == 0
            self.v = nn.Linear(dim, dim // factor ** 2)  # per child pixel, then unshuffle
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                      # x: B,H,W,C (H,W multiples of region)
        R = self.region
        if R == 0 or (x.shape[1] <= R and x.shape[2] <= R):
            return self._attend(x)
        B, H, W, C = x.shape
        if self.shift:
            x = torch.roll(x, (-(R // 2), -(R // 2)), dims=(1, 2))
        nh, nw = H // R, W // R
        r = x.view(B, nh, R, nw, R, C).permute(0, 1, 3, 2, 4, 5).reshape(B * nh * nw, R, R, C)
        o = self._attend(r).view(B, nh, nw, R, R, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        if self.shift:
            o = torch.roll(o, (R // 2, R // 2), dims=(1, 2))
        return o

    def _attend(self, x):                      # x: B,H,W,C (H,W multiples of factor)
        B, H, W, C = x.shape
        f, h = self.f, self.h
        xc = x.permute(0, 3, 1, 2)                                               # B,C,H,W
        coarse = F.avg_pool2d(xc, f).permute(0, 2, 3, 1)                         # B,h',w',C
        q = self.q(x).reshape(B, H * W, h, C // h).transpose(1, 2)               # B,h,Nq,d
        k = self.k(coarse).reshape(B, -1, h, C // h).transpose(1, 2)             # B,h,Nk,d
        if self.mode == "pool":
            v = self.v(coarse)
        else:
            # child block of every coarse location: B, C/f^2 * f^2 = C, h', w'
            v = F.pixel_unshuffle(self.v(x).permute(0, 3, 1, 2), f).permute(0, 2, 3, 1)
        v = v.reshape(B, -1, h, C // h).transpose(1, 2).contiguous()             # B,h,Nk,d
        q, k = q.contiguous(), k.contiguous()  # fused kernels need stride(-1) == 1
        nq = q.shape[2]
        if nq <= self.q_chunk:
            o = F.scaled_dot_product_attention(q, k, v)
        else:   # bounded memory for large test images
            o = torch.cat([F.scaled_dot_product_attention(q[:, :, i:i + self.q_chunk], k, v)
                           for i in range(0, nq, self.q_chunk)], dim=2)
        return self.proj(o.transpose(1, 2).reshape(B, H, W, C))


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
class ConvFFN(nn.Module):
    def __init__(self, dim, ratio=2.0):
        super().__init__()
        hid = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hid)
        self.dw = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)
        self.fc2 = nn.Linear(hid, dim)

    def forward(self, x):                      # B,H,W,C
        y = self.fc1(x)
        y = y + self.dw(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.fc2(F.gelu(y))


class Block(nn.Module):
    def __init__(self, dim, heads, window, shift, global_mode, factor, ffn_ratio,
                 region=32, region_shift=False):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.local = WindowAttention(dim, heads, window, shift)
        self.glob = None
        if global_mode != "none":
            self.ng = nn.LayerNorm(dim)
            self.glob = GlobalAttention(dim, heads, factor, global_mode, region, region_shift)
        self.n2 = nn.LayerNorm(dim)
        self.ffn = ConvFFN(dim, ffn_ratio)

    def forward(self, x):
        x = x + self.local(self.n1(x))
        if self.glob is not None:
            x = x + self.glob(self.ng(x))
        return x + self.ffn(self.n2(x))


class Group(nn.Module):
    def __init__(self, dim, depth, heads, window, global_mode, factor, ffn_ratio, global_every,
                 region=32):
        super().__init__()
        is_g = [i % global_every == global_every - 1 for i in range(depth)]
        g_idx = [sum(is_g[:i]) for i in range(depth)]      # index among global layers
        self.blocks = nn.ModuleList([
            Block(dim, heads, window, shift=(i % 2 == 1),
                  global_mode=global_mode if is_g[i] else "none",
                  factor=factor, ffn_ratio=ffn_ratio, region=region,
                  region_shift=(g_idx[i] % 2 == 1))
            for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):                      # B,C,H,W
        y = x.permute(0, 2, 3, 1)
        for b in self.blocks:
            y = b(y)
        return x + self.conv(y.permute(0, 3, 1, 2))


class CSEAT(nn.Module):
    def __init__(self, scale=4, dim=64, groups=4, depth=4, heads=4, window=8,
                 global_mode="xscale", factor=2, ffn_ratio=2.0, global_every=2, region=32,
                 img_range=1.0, rgb_mean=(0.4488, 0.4371, 0.4040)):
        super().__init__()
        self.scale, self.window, self.factor = scale, window, factor
        self.region = region if global_mode != "none" else 0
        self.register_buffer("mean", torch.tensor(rgb_mean).view(1, 3, 1, 1), persistent=False)
        self.head = nn.Conv2d(3, dim, 3, padding=1)
        self.body = nn.Sequential(*[Group(dim, depth, heads, window, global_mode, factor,
                                          ffn_ratio, global_every, region) for _ in range(groups)])
        self.body_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.tail = nn.Sequential(nn.Conv2d(dim, 3 * scale ** 2, 3, padding=1),
                                  nn.PixelShuffle(scale))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):                      # x: B,3,H,W in [0,1]
        H, W = x.shape[-2:]
        mult = self.window * self.factor // math.gcd(self.window, self.factor)
        if self.region:
            mult = mult * self.region // math.gcd(mult, self.region)
        ph, pw = (-H) % mult, (-W) % mult
        if ph or pw:
            ok = ph < H and pw < W
            x = F.pad(x, (0, pw, 0, ph), mode="reflect" if ok else "replicate")
        x = x - self.mean
        f = self.head(x)
        f = f + self.body_conv(self.body(f))
        out = self.tail(f) + self.mean
        return out[..., :H * self.scale, :W * self.scale]


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
