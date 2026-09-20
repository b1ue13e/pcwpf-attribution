$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BundleName = "cloud_4090_ae_fast_revision_bundle_v1"
$StageDir = Join-Path $RepoRoot $BundleName
$ZipPath = Join-Path $RepoRoot "$BundleName.zip"

Add-Type -AssemblyName System.IO.Compression.FileSystem

function New-CleanDirectory {
    param([string]$Path)
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
    New-Item -ItemType Directory -Path $Path | Out-Null
}

function Ensure-ParentDirectory {
    param([string]$Path)
    $TargetDir = Split-Path -Parent $Path
    if (-not (Test-Path $TargetDir)) {
        New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    }
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
    Ensure-ParentDirectory -Path $Target
    Copy-Item -Path $Source -Destination $Target -Force
}

Write-Host "[bundle] staging directory: $StageDir"
New-CleanDirectory -Path $StageDir

$FilesToCopy = @(
    @{ Source = (Join-Path $RepoRoot "wind_final.csv"); Relative = "wind_final.csv" }
    @{ Source = (Join-Path $RepoRoot "guardrail_backbones.py"); Relative = "guardrail_backbones.py" }
    @{ Source = (Join-Path $RepoRoot "guardrail_proxy_utils.py"); Relative = "guardrail_proxy_utils.py" }
    @{ Source = (Join-Path $RepoRoot "priority_dataset_hub.py"); Relative = "priority_dataset_hub.py" }
    @{ Source = (Join-Path $RepoRoot "ultra_lsnt_timeseries.py"); Relative = "ultra_lsnt_timeseries.py" }
    @{ Source = (Join-Path $RepoRoot "run_pi_moe_full_training_evidence.py"); Relative = "run_pi_moe_full_training_evidence.py" }
    @{ Source = (Join-Path $RepoRoot "run_clip_only_guardrail_control.py"); Relative = "run_clip_only_guardrail_control.py" }
    @{ Source = (Join-Path $RepoRoot "export_guardrail_prediction_archives.py"); Relative = "export_guardrail_prediction_archives.py" }
    @{ Source = (Join-Path $RepoRoot "run_dispatch_closure_mapping_decision_4090.py"); Relative = "run_dispatch_closure_mapping_decision_4090.py" }
    @{ Source = (Join-Path $RepoRoot "run_network_constrained_dispatch_ieee24.py"); Relative = "run_network_constrained_dispatch_ieee24.py" }
    @{ Source = (Join-Path $RepoRoot "dispatch_mapping_utils.py"); Relative = "dispatch_mapping_utils.py" }
    @{ Source = (Join-Path $RepoRoot "build_decision_aware_dispatch_evidence.py"); Relative = "build_decision_aware_dispatch_evidence.py" }
    @{ Source = (Join-Path $RepoRoot "wind_dispatch_model.py"); Relative = "wind_dispatch_model.py" }
    @{ Source = (Join-Path $RepoRoot "run_ae_proxy_sensitivity_case.py"); Relative = "run_ae_proxy_sensitivity_case.py" }
    @{ Source = (Join-Path $RepoRoot "aggregate_ae_proxy_sensitivity_results.py"); Relative = "aggregate_ae_proxy_sensitivity_results.py" }
    @{ Source = (Join-Path $RepoRoot "run_ae_dispatch_param_sensitivity.py"); Relative = "run_ae_dispatch_param_sensitivity.py" }
    @{ Source = (Join-Path $RepoRoot "run_ae_fast_revision_quick_smoke.sh"); Relative = "run_ae_fast_revision_quick_smoke.sh" }
    @{ Source = (Join-Path $RepoRoot "run_ae_fast_revision_full_suite_4090.sh"); Relative = "run_ae_fast_revision_full_suite_4090.sh" }
    @{ Source = (Join-Path $RepoRoot "README_cloud_ae_fast_revision.md"); Relative = "README_cloud_ae_fast_revision.md" }
    @{ Source = (Join-Path $RepoRoot "requirements_cloud_4090.txt"); Relative = "requirements_cloud_4090.txt" }
    @{ Source = (Join-Path $RepoRoot "manifests\ae_fast_revision_seed42_main_manifest.json"); Relative = "manifests/ae_fast_revision_seed42_main_manifest.json" }
    @{ Source = (Join-Path $RepoRoot "manifests\ae_fast_revision_seed42_decoupling_manifest.json"); Relative = "manifests/ae_fast_revision_seed42_decoupling_manifest.json" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\main\best_model.pth"); Relative = "checkpoints_ts/main/best_model.pth" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\main\model_config.json"); Relative = "checkpoints_ts/main/model_config.json" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\main\final_results.json"); Relative = "checkpoints_ts/main/final_results.json" }
    @{ Source = (Join-Path $RepoRoot "results\supplementary_evidence\dlinear_predictions_full.npz"); Relative = "results/supplementary_evidence/dlinear_predictions_full.npz" }
    @{ Source = (Join-Path $RepoRoot "checkpoints_ts\TimeMixer_preds.npy"); Relative = "checkpoints_ts/TimeMixer_preds.npy" }
    @{ Source = (Join-Path $RepoRoot "results\supplementary_evidence\fgmoe_clip_only_control_windcn.csv"); Relative = "results/supplementary_evidence/fgmoe_clip_only_control_windcn.csv" }
    @{ Source = (Join-Path $RepoRoot "results\supplementary_evidence\fgmoe_clip_only_control_windcn.meta.json"); Relative = "results/supplementary_evidence/fgmoe_clip_only_control_windcn.meta.json" }
    @{ Source = (Join-Path $RepoRoot "results_export_cfg_lower_corr_20260329_012420\stage1_summary_cfg_lower_corr\operation_high_risk_summary_seed42.csv"); Relative = "results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_high_risk_summary_seed42.csv" }
    @{ Source = (Join-Path $RepoRoot "results_export_cfg_lower_corr_20260329_012420\stage1_summary_cfg_lower_corr\operation_high_risk_summary_seed42.meta.json"); Relative = "results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_high_risk_summary_seed42.meta.json" }
    @{ Source = (Join-Path $RepoRoot "results_export_cfg_lower_corr_20260329_012420\stage1_summary_cfg_lower_corr\operation_decoupling_summary_seed42.csv"); Relative = "results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_decoupling_summary_seed42.csv" }
    @{ Source = (Join-Path $RepoRoot "results_export_cfg_lower_corr_20260329_012420\stage1_summary_cfg_lower_corr\operation_decoupling_summary_seed42.meta.json"); Relative = "results_export_cfg_lower_corr_20260329_012420/stage1_summary_cfg_lower_corr/operation_decoupling_summary_seed42.meta.json" }
    @{ Source = (Join-Path $RepoRoot "upload_results_ae_operation_20260328_104442\ae_operation_exports\windcn_ultra\seed_42\prediction_archive_raw_windcn_ultra_seed42.npz"); Relative = "upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_ultra/seed_42/prediction_archive_raw_windcn_ultra_seed42.npz" }
    @{ Source = (Join-Path $RepoRoot "upload_results_ae_operation_20260328_104442\ae_operation_exports\windcn_ultra\seed_42\prediction_archive_clip_only_windcn_ultra_seed42.npz"); Relative = "upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_ultra/seed_42/prediction_archive_clip_only_windcn_ultra_seed42.npz" }
    @{ Source = (Join-Path $RepoRoot "upload_results_ae_operation_20260328_104442\ae_operation_exports\windcn_ultra\seed_42\prediction_archive_full_guardrail_windcn_ultra_seed42.npz"); Relative = "upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_ultra/seed_42/prediction_archive_full_guardrail_windcn_ultra_seed42.npz" }
    @{ Source = (Join-Path $RepoRoot "upload_results_ae_operation_20260328_104442\ae_operation_exports\windcn_itransformer\seed_42\prediction_archive_raw_windcn_itransformer_seed42.npz"); Relative = "upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_itransformer/seed_42/prediction_archive_raw_windcn_itransformer_seed42.npz" }
    @{ Source = (Join-Path $RepoRoot "upload_results_ae_operation_20260328_104442\ae_operation_exports\windcn_itransformer\seed_42\prediction_archive_full_guardrail_windcn_itransformer_seed42.npz"); Relative = "upload_results_ae_operation_20260328_104442/ae_operation_exports/windcn_itransformer/seed_42/prediction_archive_full_guardrail_windcn_itransformer_seed42.npz" }
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
