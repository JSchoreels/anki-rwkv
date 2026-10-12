param([switch]$Worker)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskOutput = Join-Path $taskRoot 'artifacts/benchmarks/gpu-baseline-b01'
if (-not (Test-Path -LiteralPath (Join-Path $taskOutput 'protocol.json'))) {
    throw 'Freeze and record the authorized B01 protocol before launch.'
}
if (-not $Worker) {
    $taskProcess = Start-Process -FilePath (Get-Process -Id $PID).Path `
        -ArgumentList @('-NoProfile', '-File', ('"' + $PSCommandPath + '"'), '-Worker') `
        -WindowStyle Hidden -WorkingDirectory $taskRoot -PassThru
    Write-Output "B01 worker PID: $($taskProcess.Id); progress: artifacts/benchmarks/gpu-baseline-b01/summary.md"
    exit 0
}
Set-Location -LiteralPath $taskRoot
$taskState = Join-Path $taskOutput 'process.json'
if (Test-Path -LiteralPath $taskState) { throw 'Preserve the existing B01 launch state; no implicit restart.' }
$env:CUPY_CACHE_DIR = Join-Path $taskRoot 'artifacts/cache/cupy-cache'
$env:TEMP = Join-Path $taskRoot 'artifacts/diagnostics/check-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:CUPY_CACHE_DIR, $env:TEMP -Force | Out-Null
@{ pid = $PID; status = 'running'; started_utc = [DateTime]::UtcNow.ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath $taskState
try {
    & (Join-Path $taskRoot '.venv/Scripts/python.exe') -u scripts/run_gpu_b01.py *>> (Join-Path $taskOutput 'console.log')
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
