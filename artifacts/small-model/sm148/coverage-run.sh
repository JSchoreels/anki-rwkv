#!/usr/bin/env bash
set -u
export PATH=/usr/bin:/mingw64/bin:$PATH
S=C:/Users/jonat/workspace/wt/sm148/artifacts/small-model/sm148; CV=$S/cpu/cov
rm -f "$CV/memguard.off" "$CV/DONE"
wp=$(cat /proc/$$/winpid 2>/dev/null || echo 0)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(cygpath -w "$S/memguard.ps1")" -Dir "$(cygpath -w "$CV")" -ParentPid "$wp" \
  -Match "small_model_heldout.py&--device cpu&sm148/cpu/cov" > /dev/null 2>&1 &
MG=$!
bash "$S/coverage-lc.sh"
touch "$CV/memguard.off"; sleep 12; kill $MG 2>/dev/null
