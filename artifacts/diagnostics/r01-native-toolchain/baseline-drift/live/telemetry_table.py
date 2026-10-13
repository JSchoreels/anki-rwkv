import json, glob, os, sys, statistics as st

import os as _os
R = _os.environ.get('R01_ROOT', 'artifacts/retraining/gpu-retrain-r01/')


def jl(p):
    out = []
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def row(case):
    d = json.load(open(case + '/result.json', encoding='utf-8'))
    if d.get('smoke_only'):
        return None
    gp = [g for g in jl(case + '/gpu_power.jsonl') if (g.get('utilization.gpu') or 0) >= 20]
    rs = [r for r in jl(case + '/resources.jsonl') if 'training' in str(r.get('phase', ''))]
    up = jl(case + '/updates.jsonl')
    main = []
    workers = []
    for r in rs:
        procs = r.get('processes') or []
        if procs:
            main.append(procs[0].get('cpu_percent_one_core_100'))
            workers.append(sum(p.get('cpu_percent_one_core_100') or 0 for p in procs[1:]))
    tot = [mean(r['cpu_per_core_percent']) for r in rs if r.get('cpu_per_core_percent')]
    gpu_ms = sum(u.get('gpu_ms') or 0 for u in up)
    disp = sum(u.get('host_dispatch_s') or 0 for u in up)
    qw = sum(u.get('queue_wait_seconds') or 0 for u in up)
    f = lambda v, n=0: '—' if v is None else f'{v:.{n}f}'
    return [d.get('started_utc', '')[5:16], f(d['reviews_per_second']), f(d['training_seconds'], 2),
            f(mean(g.get('clocks.sm') for g in gp)), f(mean(g.get('clocks.mem') for g in gp)),
            f(mean(g.get('power.draw') for g in gp)), f(mean(g.get('temperature.gpu') for g in gp)),
            f(mean(g.get('utilization.gpu') for g in gp)),
            f(mean(tot)), f(mean(main)), f(mean(workers)),
            f(mean(r.get('ram_available_bytes', 0) / 2**30 for r in rs), 1),
            f(gpu_ms / 1000, 2) if up else '—', f(disp, 2) if up else '—', f(qw, 2) if up else '—', f(sum(u.get('preparation_seconds') or 0 for u in up), 1),
            str(d.get('warmup_job_workers')), case.replace(R, '').replace('/expertium', '')[:40]]


hdr = ['start', 'rps', 'train_s', 'smMHz', 'memMHz', 'W', 'C', 'gpu%', 'cpu%', 'main%', 'wrk%', 'ramGB', 'gpu_s', 'disp_s', 'qw_s', 'prep_s', 'wk', 'case']
cases = []
for pat in sys.argv[1:]:
    cases += sorted(glob.glob(R + pat))
print('\t'.join(hdr))
for c in cases:
    c = c.replace(os.sep, '/')
    try:
        r = row(c)
    except Exception as e:
        r = ['ERR', str(e)[:60], c]
    if r:
        print('\t'.join(r))
