import json, sys

R = 'artifacts/retraining/gpu-retrain-r01/'
for c in sys.argv[1:]:
    rs = [json.loads(l) for l in open(R + c + '/resources.jsonl', encoding='utf-8')]
    first = {}
    last = {}
    phases = []
    for r in rs:
        if r.get('phase') not in phases:
            phases.append(r.get('phase'))
        if 'training' not in str(r.get('phase')):
            continue
        for p in r.get('processes') or []:
            first.setdefault(p['pid'], p)
            last[p['pid']] = p
    print(c.split('/')[2] if 'replicates' in c else c, '| phases:', phases[:6])
    for pid, p in last.items():
        a = first[pid]
        print(f"  {p['name']:12s} ppid={p['parent_pid']} pf+={(p['page_faults']-a['page_faults'])/1e3:8.0f}k ioR+={(p['io_read_bytes']-a['io_read_bytes'])/2**20:7.0f}MiB ioW+={(p['io_write_bytes']-a['io_write_bytes'])/2**20:6.0f}MiB rss={p['rss_bytes']/2**20:6.0f}MiB")
    tr = [r for r in rs if 'training' in str(r.get('phase'))]
    cores = [sum(x) / len(tr) for x in zip(*[r['cpu_per_core_percent'] for r in tr])]
    print('  per-core mean%:', ' '.join(f'{x:.0f}' for x in cores))
