param([switch]$Worker)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'artifacts/benchmarks/gpu-baseline-b01-deterministic-v3'
$taskProtocol = Join-Path $taskOutput 'protocol.json'
if (-not (Test-Path -LiteralPath $taskProtocol)) { throw 'Freeze the separate B01 retry protocol before launch.' }
$taskFrozen = Get-Content -LiteralPath $taskProtocol -Raw | ConvertFrom-Json
if (-not $taskFrozen.execution_mode.deterministic_algorithms -or $taskFrozen.execution_mode.CUBLAS_WORKSPACE_CONFIG -ne ':4096:8') {
    throw 'Expected the frozen deterministic execution configuration.'
}
if (-not $Worker) {
    $taskProcess = Start-Process -FilePath (Get-Process -Id $PID).Path `
        -ArgumentList @('-NoProfile', '-File', ('"' + $PSCommandPath + '"'), '-Worker') `
        -WindowStyle Hidden -WorkingDirectory $taskRoot -PassThru
    Write-Output "B01 retry worker PID: $($taskProcess.Id); progress: artifacts/benchmarks/gpu-baseline-b01-deterministic-v3/summary.md"
    exit 0
}
Set-Location -LiteralPath $taskRoot
$taskState = Join-Path $taskOutput 'process.json'
if (Test-Path -LiteralPath $taskState) { throw 'Preserve an existing retry launch state; no implicit restart.' }
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:CUPY_CACHE_DIR = Join-Path $taskRoot 'artifacts/cache/cupy-cache'
$env:TEMP = Join-Path $taskRoot 'artifacts/diagnostics/check-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:CUPY_CACHE_DIR, $env:TEMP -Force | Out-Null
@{ pid = $PID; status = 'running'; started_utc = [DateTime]::UtcNow.ToString('o'); CUBLAS_WORKSPACE_CONFIG = $env:CUBLAS_WORKSPACE_CONFIG } |
    ConvertTo-Json | Set-Content -LiteralPath $taskState
try {
    & (Join-Path $taskRoot '.venv/Scripts/python.exe') -u scripts/run_gpu_b01.py --output artifacts/benchmarks/gpu-baseline-b01-deterministic-v3 *>> (Join-Path $taskOutput 'console.log')
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
