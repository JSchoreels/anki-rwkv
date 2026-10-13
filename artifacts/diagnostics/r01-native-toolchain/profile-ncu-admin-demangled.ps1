$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\jonat\workspace\anki-rwkv'
$taskOutput = Join-Path $taskRoot 'artifacts\diagnostics\r01-native-toolchain\ncu-administrator-demangled-launch.json'
$taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$taskPrincipal = New-Object Security.Principal.WindowsPrincipal($taskIdentity)
if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator context required' }
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
@{ status = 'running'; administrator = $true; started_utc = [DateTime]::UtcNow.ToString('o'); authorization = 'Install Build Tools and allow elevated profiling'; selection = 'One backward_kernel, demangled name base, inside strict R01_T256_backward NVTX range; repeats zero-capture probe after repairing stripped-name filter'; security_settings_changed = $false } | ConvertTo-Json | Set-Content -LiteralPath $taskOutput -Encoding UTF8
& '.venv\Scripts\python.exe' '-X' 'utf8' 'scripts\r01_preflight.py' '--output' 'artifacts\diagnostics\r01-native-toolchain\preflight-admin-demangled.json' '--main-threads' '1' '--run' 'scripts\check_r01_nsight.py' '--output' 'artifacts\diagnostics\r01-native-toolchain\ncu-administrator-demangled' '--tool' 'ncu' *> (Join-Path $taskRoot 'artifacts\diagnostics\r01-native-toolchain\ncu-administrator-demangled-console.log')
$taskExit = $LASTEXITCODE
@{ status = 'exited'; administrator = $true; exit_code = $taskExit; completed_utc = [DateTime]::UtcNow.ToString('o'); security_settings_changed = $false } | ConvertTo-Json | Set-Content -LiteralPath $taskOutput -Encoding UTF8
exit $taskExit

