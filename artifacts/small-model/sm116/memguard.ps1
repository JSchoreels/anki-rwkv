# SM116 memory guard: while the scoring pass runs, every 10 s, log the working set of each SM116 scorer process to
# rows/memguard.log and kill any above 5.9 GB (the 6 GB per-process limit); a kill writes rows/STOP and rows/memguard-kill.
$dir = 'C:\Users\jonat\workspace\wt\sm116\artifacts\small-model\sm116\rows'
while (-not (Test-Path "$dir\memguard.off")) {
  $t = Get-Date -Format 'MM-dd HH:mm:ss'
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*sm116_state.py score*' } | ForEach-Object {
    $gb = [math]::Round($_.WorkingSetSize / 1GB, 2)
    if ($gb -gt 1) { Add-Content "$dir\memguard.log" "$t $($_.ProcessId) $gb" }
    if ($gb -gt 5.9) {
      Stop-Process -Id $_.ProcessId -Force
      Add-Content "$dir\memguard-kill" "$t killed $($_.ProcessId) at $gb GB: $($_.CommandLine.Substring([math]::Max(0, $_.CommandLine.Length - 60)))"
      New-Item -ItemType File -Force "$dir\STOP" | Out-Null
    }
  }
  Start-Sleep 10
}
