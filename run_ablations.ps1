# Phase 1: equal-budget ablation (50k iters, bs16, patch 64) - run sequentially.
# Resumable: re-running this script continues any unfinished experiment from last.pt.
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }   # set $env:PYTHON to use a specific interpreter
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
foreach ($mode in "xscale", "pool", "none") {
    $exp = "abl50k_$mode"
    New-Item -ItemType Directory -Force "experiments\$exp" | Out-Null
    if (Test-Path "experiments\$exp\final.pt") { Write-Output "skip $exp (done)"; continue }
    Write-Output "=== $exp start $(Get-Date -Format s) ==="
    & $py -u -W ignore train.py --exp $exp --global-mode $mode --iters 50000 --warmup 1000 *>> "experiments\$exp\train.log"
    & $py -u -W ignore test.py --ckpt "experiments\$exp\final.pt" *>> "experiments\$exp\test.log"
    Write-Output "=== $exp end $(Get-Date -Format s) ==="
}
