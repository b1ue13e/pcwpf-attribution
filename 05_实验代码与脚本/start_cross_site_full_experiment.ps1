param(
    [string]$OutDir = "",
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Script = Join-Path $PSScriptRoot "run_cross_site_recalibration_experiments.py"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $ResultsRoot = Get-ChildItem -Path $ProjectRoot -Directory | Where-Object { $_.Name -like "06_*" } | Select-Object -First 1
    if ($null -eq $ResultsRoot) {
        throw "Cannot find results root directory under $ProjectRoot"
    }
    $RunDir = Join-Path $ResultsRoot.FullName "results\cross_site_wind_full_cuda"
} elseif ([System.IO.Path]::IsPathRooted($OutDir)) {
    $RunDir = $OutDir
} else {
    $RunDir = Join-Path $ProjectRoot $OutDir
}
$LogDir = Join-Path $RunDir "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $LogDir "full_cuda_$stamp.out.log"
$stderr = Join-Path $LogDir "full_cuda_$stamp.err.log"
$pidFile = Join-Path $RunDir "full_cuda.pid"

$arguments = @(
    $Script,
    "--device", "cuda",
    "--models", "ultra_lsnt,dlinear,itransformer",
    "--datasets", "wind_cn_core_15min,wind_us_core_15min,kelmarsh_uk_2016,penmanshiel_scotland_2016,smarteole_france_2020",
    "--seeds", "42,123,2025",
    "--calibration-ratios", "0,0.01,0.05,0.10,0.50,1.0",
    "--batch-size", "$BatchSize",
    "--seq-len", "96",
    "--pred-len", "24",
    "--epochs", "20",
    "--recalibration-epochs", "8",
    "--patience", "5",
    "--recalibration-patience", "3",
    "--out-dir", $RunDir,
    "--resume"
)

$proc = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii

[pscustomobject]@{
    Pid = $proc.Id
    OutDir = $RunDir
    Stdout = $stdout
    Stderr = $stderr
    PidFile = $pidFile
}
