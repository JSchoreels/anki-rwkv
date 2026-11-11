param([string]$BaseExe, [string]$NewExe, [int]$Rounds = 4, [string]$Runtime = "small")
# Alternating-order A/B (new first on odd rounds) of one runtime, one thread.
$env:ANKI_RWKV_SIZE_BENCH_RUNTIMES = $Runtime
$env:ANKI_RWKV_SIZE_BENCH_SAMPLES = "5"
$env:RAYON_NUM_THREADS = "1"
for ($round = 1; $round -le $Rounds; $round++) {
    $order = if ($round % 2 -eq 1) { @("new", "base") } else { @("base", "new") }
    foreach ($tree in $order) {
        $exe = if ($tree -eq "base") { $BaseExe } else { $NewExe }
        $log = [System.IO.Path]::GetTempFileName()
        $p = Start-Process -FilePath $exe -ArgumentList @("rwkv_model_size_benchmark", "--ignored", "--nocapture", "--test-threads", "1") -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        $null = $p.Handle
        $p.WaitForExit()
        $line = (Get-Content $log | Where-Object { $_ -like "${Runtime}: *" }) -join " "
        Write-Output ("{0} {1,-5} cpu {2,6:N2}s | {3}" -f $round, $tree, $p.TotalProcessorTime.TotalSeconds, $line)
        Remove-Item $log, "$log.err" -ErrorAction SilentlyContinue
    }
}
