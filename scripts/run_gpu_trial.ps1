param(
    [switch]$Resume,
    [switch]$Check,
    [switch]$Worker
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$taskPrincipal = [Security.Principal.WindowsPrincipal]::new($taskIdentity)
if (-not $Worker) {
    $taskArguments = @('-NoProfile', '-File', ('"' + $PSCommandPath + '"'))
    $taskArguments += '-Worker'
    if ($Resume) { $taskArguments += '-Resume' }
    if ($Check) { $taskArguments += '-Check' }
    $taskLaunch = @{ FilePath = (Get-Process -Id $PID).Path; ArgumentList = $taskArguments;
        WindowStyle = 'Hidden'; WorkingDirectory = $taskRoot; PassThru = $true }
    $taskConfig = Get-Content -LiteralPath (Join-Path $taskRoot 'configs/train-gpu-trial.toml') -Raw
    if ($taskConfig -match '(?m)^POWER_LIMIT_WATTS\s*=\s*([0-9]+)' -and [int]$Matches[1] -gt 0 `
        -and -not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        $taskLaunch.Verb = 'RunAs'
    }
    $taskProcess = Start-Process @taskLaunch
    Write-Output "GPU job PID: $($taskProcess.Id). Its log is in artifacts/benchmarks/gpu-trial or artifacts/diagnostics/gpu-smoke."
    exit 0
}

Set-Location -LiteralPath $taskRoot
$taskOutput = if ($Check) { Join-Path $taskRoot 'artifacts/diagnostics/gpu-smoke' } else { Join-Path $taskRoot 'artifacts/benchmarks/gpu-trial' }
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$taskStatePath = Join-Path $taskOutput 'process.json'
if (Test-Path -LiteralPath $taskStatePath) {
    $taskPrevious = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json
    if ($taskPrevious.status -eq 'running' -and (Get-Process -Id $taskPrevious.pid -ErrorAction SilentlyContinue)) {
        throw "GPU job $($taskPrevious.pid) is already running."
    }
}
$env:CUPY_CACHE_DIR = Join-Path $taskRoot 'artifacts/cache/cupy-cache'
$env:TEMP = Join-Path $taskRoot 'artifacts/diagnostics/check-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:CUPY_CACHE_DIR, $env:TEMP -Force | Out-Null
$taskLog = Join-Path $taskOutput 'console.log'
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
$taskScript = if ($Check) { 'scripts/check_gpu.py' } else { 'scripts/train_gpu_trial.py' }
$taskParameters = @('-u', $taskScript)
if ($Resume -and -not $Check) { $taskParameters += '--resume' }

@{ pid = $PID; status = 'running'; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'process.json')
try {
    & $taskPython @taskParameters *>> $taskLog
    $taskCode = $LASTEXITCODE
    @{ pid = $PID; status = if ($taskCode -eq 0) { 'exited' } else { 'failed' }; exit_code = $taskCode } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'process.json')
    exit $taskCode
} catch {
    $_ | Out-String | Add-Content -LiteralPath $taskLog
    @{ pid = $PID; status = 'failed'; error = $_.Exception.Message } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOutput 'process.json')
    exit 1
}
