$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleName = "cloud_4090_missing_ae_bundle_v1"
$StageDir = Join-Path $RepoRoot $BundleName
$ZipPath = Join-Path $RepoRoot "$BundleName.zip"

function New-CleanDirectory {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Copy-BundleFile {
    param(
        [string]$Source,
        [string]$RelativeTarget
    )
    if (-not (Test-Path $Source)) {
        throw "Missing required source file: $Source"
    }
    $Target = Join-Path $StageDir $RelativeTarget
    $TargetDir = Split-Path -Parent $Target
    if (-not (Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }
    Copy-Item -Path $Source -Destination $Target -Force
}

Write-Host "[bundle] staging directory: $StageDir"
New-CleanDirectory -Path $StageDir

$FilesToCopy = @(
    @{ Source = (Join-Path $RepoRoot "wind_final.csv"); Relative = "wind_final.csv" }
    @{ Source = (Join-Path $RepoRoot "wind_us.csv"); Relative = "wind_us.csv" }
    @{ Source = (Join-Path $RepoRoot "guardrail_backbones.py"); Relative = "guardrail_backbones.py" }
    @{ Source = (Join-Path $RepoRoot "guardrail_proxy_utils.py"); Relative = "guardrail_proxy_utils.py" }
    @{ Source = (Join-Path $RepoRoot "priority_dataset_hub.py"); Relative = "priority_dataset_hub.py" }
    @{ Source = (Join-Path $RepoRoot "ultra_lsnt_timeseries.py"); Relative = "ultra_lsnt_timeseries.py" }
    @{ Source = (Join-Path $RepoRoot "run_pi_moe_full_training_evidence.py"); Relative = "run_pi_moe_full_training_evidence.py" }
    @{ Source = (Join-Path $RepoRoot "run_clip_only_guardrail_control.py"); Relative = "run_clip_only_guardrail_control.py" }
    @{ Source = (Join-Path $RepoRoot "run_safety_deployment.py"); Relative = "run_safety_deployment.py" }
    @{ Source = (Join-Path $RepoRoot "aggregate_missing_ae_results.py"); Relative = "aggregate_missing_ae_results.py" }
    @{ Source = (Join-Path $RepoRoot "run_missing_ae_quick_smoke.sh"); Relative = "run_missing_ae_quick_smoke.sh" }
    @{ Source = (Join-Path $RepoRoot "run_missing_ae_full_suite_4090.sh"); Relative = "run_missing_ae_full_suite_4090.sh" }
    @{ Source = (Join-Path $RepoRoot "README_cloud_missing_ae.md"); Relative = "README_cloud_missing_ae.md" }
    @{ Source = (Join-Path $RepoRoot "requirements_cloud_4090.txt"); Relative = "requirements_cloud_4090.txt" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\main\best_model.pth"); Relative = "checkpoints_ts/main/best_model.pth" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\main\model_config.json"); Relative = "checkpoints_ts/main/model_config.json" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\main\final_results.json"); Relative = "checkpoints_ts/main/final_results.json" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\exp_wind_us\best_model.pth"); Relative = "checkpoints_ts/exp_wind_us/best_model.pth" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\exp_wind_us\model_config.json"); Relative = "checkpoints_ts/exp_wind_us/model_config.json" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\exp_wind_us\final_results.json"); Relative = "checkpoints_ts/exp_wind_us/final_results.json" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\iTransformer_best.pth"); Relative = "checkpoints_ts/iTransformer_best.pth" }
)

foreach ($Item in $FilesToCopy) {
    Copy-BundleFile -Source $Item.Source -RelativeTarget $Item.Relative
}

$ManifestPath = Join-Path $StageDir "bundle_manifest.txt"
$ShaPath = Join-Path $StageDir "SHA256SUMS.txt"
Set-Content -Path $ManifestPath -Value "" -Encoding UTF8
Set-Content -Path $ShaPath -Value "" -Encoding UTF8

$RelativeFiles = Get-ChildItem -Path $StageDir -Recurse -File |
    Where-Object { $_.FullName -ne $ShaPath } |
    Sort-Object FullName |
    ForEach-Object {
        $_.FullName.Substring($StageDir.Length + 1).Replace("\", "/")
    }

Set-Content -Path $ManifestPath -Value ($RelativeFiles -join [Environment]::NewLine) -Encoding UTF8

$HashLines = Get-ChildItem -Path $StageDir -Recurse -File |
    Where-Object { $_.FullName -ne $ShaPath } |
    Sort-Object FullName |
    ForEach-Object {
        $Hash = (Get-FileHash -Path $_.FullName -Algorithm SHA256).Hash.ToLower()
        $Relative = $_.FullName.Substring($StageDir.Length + 1).Replace("\", "/")
        "$Hash  $Relative"
    }

Set-Content -Path $ShaPath -Value ($HashLines -join [Environment]::NewLine) -Encoding UTF8

if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}

Write-Host "[bundle] creating zip: $ZipPath"
Compress-Archive -Path $StageDir -DestinationPath $ZipPath -CompressionLevel Optimal -Force

$ZipInfo = Get-Item $ZipPath
Write-Host ("[bundle] done: {0} ({1:N0} bytes)" -f $ZipInfo.FullName, $ZipInfo.Length)
