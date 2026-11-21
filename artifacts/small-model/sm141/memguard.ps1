# Memory guard of the tiered card state by answer count (SM141): a copy of the trained smaller state for Tiny (SM137)'s memguard.ps1.
# Every 10 s, log the working set of every python.exe whose command line contains ALL words of one -Match group (groups
# separated by ';', words by '&'), and kill any above -LimitGB (the 6 GB per-CPU-process limit); a kill writes
# <Dir>\memguard-kill and <Dir>\STOP. Ends when <Dir>\memguard.off exists or the -ParentPid process (the launcher's Windows
# PID) is gone. GPU tickets are never matched (the groups name CPU-only commands).
param([string]$Dir, [string]$Match, [double]$LimitGB = 5.9, [int]$ParentPid = 0)
$groups = $Match.Split(';') | ForEach-Object { , ($_.Split('&')) }
while (-not (Test-Path "$Dir\memguard.off")) {
  if ($ParentPid -gt 0 -and $null -eq (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)) { break }
  $t = Get-Date -Format 'MM-dd HH:mm:ss'
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $cl = $_.CommandLine
    if ($null -eq $cl) { return }
    $hit = $false
    foreach ($g in $groups) { $all = $true; foreach ($w in $g) { if (-not $cl.Contains($w)) { $all = $false } }; if ($all) { $hit = $true } }
    if (-not $hit) { return }
    $gb = [math]::Round($_.WorkingSetSize / 1GB, 2)
    if ($gb -gt 1) { Add-Content "$Dir\memguard.log" "$t $($_.ProcessId) $gb" }
    if ($gb -gt $LimitGB) {
      Stop-Process -Id $_.ProcessId -Force
      Add-Content "$Dir\memguard-kill" "$t killed $($_.ProcessId) at $gb GB: $($cl.Substring([math]::Max(0, $cl.Length - 80)))"
      New-Item -ItemType File -Force "$Dir\STOP" | Out-Null
    }
  }
  Start-Sleep 10
}
