param(
    [string]$Tasks = "multi_domain,multi_domain_baselines,kernel_ablation,noise20,dm_extended,paper_digest,lite_suite",
    [string]$LogDir = "logs_background",
    [switch]$ContinueOnError,
    [switch]$LiteOnly,
    [switch]$DryRun,
    [string]$RemoteHost = "",
    [string]$RemoteUser = "",
    [string]$RemoteWorkDir = "",
    [string]$RemotePython = "python3"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if ($LiteOnly) {
    $Tasks = "lite_suite"
}

if (![string]::IsNullOrWhiteSpace($RemoteHost)) {
    if ([string]::IsNullOrWhiteSpace($RemoteWorkDir)) {
        throw "Remote mode requires -RemoteWorkDir."
    }

    $target = if ([string]::IsNullOrWhiteSpace($RemoteUser)) { $RemoteHost } else { "$RemoteUser@$RemoteHost" }
    $continueFlag = if ($ContinueOnError) { "1" } else { "0" }
    $dryRunFlag = if ($DryRun) { "1" } else { "0" }

    $remoteScript = @'
set -e
REMOTE_WORKDIR="$1"
TASKS="$2"
LOGDIR="$3"
CONTINUE_ON_ERROR="$4"
DRY_RUN="$5"
PYTHON_BIN="$6"

cd "$REMOTE_WORKDIR"
mkdir -p "$LOGDIR"

TS=$(date +%Y%m%d_%H%M%S)
OUT="$LOGDIR/suite_${TS}.out.log"
ERR="$LOGDIR/suite_${TS}.err.log"
PID_FILE="$LOGDIR/suite_${TS}.pid"

ARGS=(run_background_suite.py --tasks "$TASKS" --log-dir "$LOGDIR")
if [ "$CONTINUE_ON_ERROR" = "1" ]; then ARGS+=(--continue-on-error); fi
if [ "$DRY_RUN" = "1" ]; then ARGS+=(--dry-run); fi

nohup "$PYTHON_BIN" "${ARGS[@]}" >"$OUT" 2>"$ERR" < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

echo "Background suite started on remote host."
echo "PID: $PID"
echo "STDOUT: $OUT"
echo "STDERR: $ERR"
echo "PID file: $PID_FILE"
echo "Tasks: $TASKS"
'@

    $sshArgs = @(
        $target,
        "bash",
        "-s",
        "--",
        $RemoteWorkDir,
        $Tasks,
        $LogDir,
        $continueFlag,
        $dryRunFlag,
        $RemotePython
    )

    $remoteScript | & ssh @sshArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Remote startup failed with exit code $LASTEXITCODE"
    }
    exit 0
}

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdoutLog = Join-Path $LogDir "suite_${timestamp}.out.log"
$stderrLog = Join-Path $LogDir "suite_${timestamp}.err.log"
$pidFile = Join-Path $LogDir "suite_${timestamp}.pid"

$pythonCmd = "python"
try {
    $null = Get-Command $pythonCmd -ErrorAction Stop
}
catch {
    $pythonCmd = "py"
}

$args = @(
    "run_background_suite.py",
    "--tasks", $Tasks,
    "--log-dir", $LogDir
)

if ($ContinueOnError) {
    $args += "--continue-on-error"
}
if ($DryRun) {
    $args += "--dry-run"
}

$proc = Start-Process `
    -FilePath $pythonCmd `
    -ArgumentList $args `
    -WorkingDirectory $root `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru `
    -WindowStyle Hidden

$proc.Id | Out-File -FilePath $pidFile -Encoding ascii

Write-Host "Background suite started."
Write-Host "PID: $($proc.Id)"
Write-Host "STDOUT: $stdoutLog"
Write-Host "STDERR: $stderrLog"
Write-Host "PID file: $pidFile"
Write-Host "Tasks: $Tasks"
