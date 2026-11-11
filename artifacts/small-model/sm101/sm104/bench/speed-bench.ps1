param(
    [string]$BaseExe,
    [string]$NewExe,
    [int]$Rounds = 3,
    [string]$Threads = "",
    [string]$Out
)
# Interleaved A/B runs of rwkv_model_size_benchmark: one process per (tree, runtime), CPU time from the OS.
$configs = @(
    @{ tree = "base"; name = "small" }, @{ tree = "new"; name = "small" },
    @{ tree = "base"; name = "small-pair" }, @{ tree = "new"; name = "small-pair" },
    @{ tree = "base"; name = "small-instant" }, @{ tree = "new"; name = "small-instant" },
    @{ tree = "new"; name = "tiny-pair" }, @{ tree = "new"; name = "tiny-instant" },
    @{ tree = "base"; name = "big" }, @{ tree = "new"; name = "big" }
)
$results = @()
for ($round = 1; $round -le $Rounds; $round++) {
    foreach ($c in $configs) {
        $exe = if ($c.tree -eq "base") { $BaseExe } else { $NewExe }
        $env:ANKI_RWKV_SIZE_BENCH_RUNTIMES = $c.name
        $env:ANKI_RWKV_SIZE_BENCH_SAMPLES = "5"
        if ($Threads) { $env:RAYON_NUM_THREADS = $Threads } else { Remove-Item Env:RAYON_NUM_THREADS -ErrorAction SilentlyContinue }
        $log = [System.IO.Path]::GetTempFileName()
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath $exe -ArgumentList @("rwkv_model_size_benchmark", "--ignored", "--nocapture", "--test-threads", "1") -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
        $null = $p.Handle
        $p.WaitForExit()
        $wall = $sw.Elapsed.TotalSeconds
        $cpu = $p.TotalProcessorTime.TotalSeconds
        $line = (Get-Content $log | Where-Object { $_ -like "$($c.name): *" }) -join " "
        $results += [pscustomobject]@{ round = $round; tree = $c.tree; runtime = $c.name; threads = $Threads; wall_s = [math]::Round($wall, 2); cpu_s = [math]::Round($cpu, 2); line = $line }
        Write-Output ("{0} {1,-5} {2,-14} wall {3,6:N2}s cpu {4,6:N2}s | {5}" -f $round, $c.tree, $c.name, $wall, $cpu, $line)
        Remove-Item $log, "$log.err" -ErrorAction SilentlyContinue
    }
}
$results | ConvertTo-Json -Depth 3 | Out-File -Encoding utf8 $Out
