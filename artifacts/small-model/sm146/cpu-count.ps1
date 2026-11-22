# Aggressive state tiers for Tiny (SM146): count the CPU scoring processes on the whole machine (any experiment): python.exe
# whose command line has small_model_heldout.py and "--device cpu", without the venv launcher's child processes (a process
# whose parent matches too is not counted again). Prints one integer. Read only.
$p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*small_model_heldout.py*' -and $_.CommandLine -like '*--device cpu*' })
$ids = @($p | ForEach-Object { $_.ProcessId })
@($p | Where-Object { $ids -notcontains $_.ParentProcessId }).Count
