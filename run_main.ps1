# Phase 2: main CSEAT model (xscale, R=32 training), 300k iters, bs16, patch 64. Resumable from last.pt.
# Then: standard test, test-time region 64, and self-ensemble (+) test.
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # set $env:PYTHON to use a specific interpreter
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$exp = "main"
New-Item -ItemType Directory -Force "experiments\$exp" | Out-Null
Write-Output "=== $exp start $(Get-Date -Format s) ==="
if (-not (Test-Path "experiments\$exp\final.pt")) {
    & $py -u -W ignore train.py --exp $exp --global-mode xscale --iters 300000 --warmup 2000 *>> "experiments\$exp\train.log"
}
& $py -u -W ignore test.py --ckpt "experiments\$exp\final.pt" *>> "experiments\$exp\test.log"
& $py -u -W ignore test.py --ckpt "experiments\$exp\final.pt" --region 64 *>> "experiments\$exp\test.log"
Write-Output "=== $exp end $(Get-Date -Format s) ==="
