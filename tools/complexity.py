"""Params / MACs @720p / train speed for each variant."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from csea.model import CSEAT, count_params

for mode in ("none", "pool", "xscale"):
    net = CSEAT(global_mode=mode).cuda()
    x = torch.rand(1, 3, 180, 320, device="cuda")
    with torch.no_grad():
        fc = FlopCounterMode(display=False)
        with fc:
            net(x)
    opt = torch.optim.Adam(net.parameters())
    xb, yb = torch.rand(16, 3, 64, 64, device="cuda"), torch.rand(16, 3, 256, 256, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(25):
        if i == 5: torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.l1_loss(net(xb).float(), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    print(f"{mode:6s} params {count_params(net)/1e3:6.1f}K | MACs@720p {fc.get_total_flops()/2e9:6.1f}G | "
          f"train {20/(time.time()-t):.2f} it/s (bs16) | peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
    del net, opt
