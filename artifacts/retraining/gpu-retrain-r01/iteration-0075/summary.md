# R01 iteration 0075: setup failure (Windows path length), no training

Candidate `E33-F48-F50-F65b-F71-F73a` (accepted 0072 options + `host_placement: "physical"` + `preparation_workers: 5`) was frozen in [protocol.json](protocol.json). Its functional smoke failed while creating the worker-ready directory inside the isolated runtime (`runtime-0075-8d946f7a48d5`): `[WinError 206] The filename or extension is too long` — the nested runtime path reached the Windows 260-character limit (`LongPathsEnabled` is 0 on this machine). No training, timing or evaluation ran; see `../iteration-0075-expertium-smoke-console.log`.

The same options continue as [iteration 0076](../iteration-0076/summary.md) under the path-length alias `E72` (accepted 0072 configuration: E33, F48, F50, F65b), candidate `E72-F71-F73a`. Enabling Windows long paths would remove this failure mode for all sessions; that is a system setting reserved to the human.
