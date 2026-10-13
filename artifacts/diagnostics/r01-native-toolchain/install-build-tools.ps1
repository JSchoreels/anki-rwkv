$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\jonat\workspace\anki-rwkv\artifacts\diagnostics\r01-native-toolchain'
$taskInstaller = Join-Path $taskRoot 'vs_BuildTools.exe'
$taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$taskPrincipal = New-Object Security.Principal.WindowsPrincipal($taskIdentity)
if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) { throw 'Administrator context required' }
$taskSignature = Get-AuthenticodeSignature -LiteralPath $taskInstaller
if ($taskSignature.Status -ne 'Valid' -or $taskSignature.SignerCertificate.Subject -notlike '*Microsoft Corporation*') { throw 'Microsoft signature verification failed' }
@{ status = 'installing'; started_utc = [DateTime]::UtcNow.ToString('o'); administrator = $true; authorization = 'Install Build Tools and allow elevated profiling'; installer_sha256 = (Get-FileHash -LiteralPath $taskInstaller -Algorithm SHA256).Hash } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'installation-attempt-2.json') -Encoding UTF8
$taskArgs = '--quiet --wait --norestart --installPath "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools" --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.26100'
try {
    $taskProcess = Start-Process -FilePath $taskInstaller -WindowStyle Hidden -ArgumentList $taskArgs -Wait -PassThru
    $taskResult = @{ status = 'installer exited'; exit_code = $taskProcess.ExitCode; completed_utc = [DateTime]::UtcNow.ToString('o'); administrator = $true; install_path = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools' }
    $taskResult | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'installation-attempt-2-result.json') -Encoding UTF8
    if ($taskProcess.ExitCode -notin @(0, 3010)) { throw ('Installer failed with exit ' + $taskProcess.ExitCode) }
} catch {
    @{ status = 'failed'; error = $_.Exception.Message; completed_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot 'installation-attempt-2-error.json') -Encoding UTF8
    exit 1
}
