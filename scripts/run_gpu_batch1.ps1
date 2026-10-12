param(
    [switch]$Worker,
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'artifacts/benchmarks/gpu-batch1-500'
$taskState = Join-Path $taskOutput 'process.json'
if (-not (Test-Path -LiteralPath (Join-Path $taskOutput 'protocol.json'))) {
    throw 'Prepare and record the protocol before launching the approved batch.'
}
if (-not $Worker) {
    $taskArguments = @('-NoProfile', '-File', ('"' + $PSCommandPath + '"'), '-Worker')
    if ($Resume) { $taskArguments += '-Resume' }
    $taskProcess = Start-Process -FilePath (Get-Process -Id $PID).Path -ArgumentList $taskArguments `
        -WindowStyle Hidden -WorkingDirectory $taskRoot -PassThru
    Write-Output "GPU batch worker PID: $($taskProcess.Id); progress: artifacts/benchmarks/gpu-batch1-500/summary.md"
    exit 0
}

Set-Location -LiteralPath $taskRoot
if (Test-Path -LiteralPath $taskState) {
    $taskPrevious = Get-Content -LiteralPath $taskState -Raw | ConvertFrom-Json
    if ($taskPrevious.status -eq 'running' -and (Get-Process -Id $taskPrevious.pid -ErrorAction SilentlyContinue)) {
        throw "GPU batch worker $($taskPrevious.pid) is already active."
    }
}
$env:CUPY_CACHE_DIR = Join-Path $taskRoot 'artifacts/cache/cupy-cache'
$env:TEMP = Join-Path $taskRoot 'artifacts/diagnostics/check-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:CUPY_CACHE_DIR, $env:TEMP -Force | Out-Null
$taskArguments = @('-u', 'scripts/run_gpu_batch1.py', '--output', 'artifacts/benchmarks/gpu-batch1-500')
if ($Resume) { $taskArguments += '--resume' }
@{ pid = $PID; status = 'running'; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $taskState
try {
    & (Join-Path $taskRoot '.venv/Scripts/python.exe') @taskArguments *>> (Join-Path $taskOutput 'console.log')
    $taskExit = $LASTEXITCODE
    @{ pid = $PID; status = if ($taskExit -eq 0) { 'exited' } else { 'failed' }; exit_code = $taskExit } |
        ConvertTo-Json | Set-Content -LiteralPath $taskState
    exit $taskExit
} catch {
    $_ | Out-String | Add-Content -LiteralPath (Join-Path $taskOutput 'console.log')
    @{ pid = $PID; status = 'failed'; error = $_.Exception.Message } |
        ConvertTo-Json | Set-Content -LiteralPath $taskState
    exit 1
}
