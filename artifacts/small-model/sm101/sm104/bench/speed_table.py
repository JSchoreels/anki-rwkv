import json
import re
import statistics
import sys

for path in sys.argv[1:]:
    rows = json.load(open(path, encoding='utf-8-sig'))
    print(path)
    groups = {}
    for r in rows:
        m = dict(re.findall(r'(\w+)=([\d.]+)', r['line']))
        groups.setdefault((r['runtime'], r['tree']), []).append((float(m['warm_up_ms']), float(m['reviews_per_s']), r['cpu_s'], r['wall_s'], float(m['scoring_ms']), float(m['full_500_ms']), m['per_card_and_note']))
    print('| runtime | tree | warm-up ms (median of 3 runs) | reviews/s | process CPU s | wall s | queue scoring ms | 500 full ms | B per card+note |')
    for (rt, tree), v in groups.items():
        med = lambda i: statistics.median(x[i] for x in v)
        print(f'| {rt} | {tree} | {med(0):.0f} | {med(1):.0f} | {med(2):.1f} | {med(3):.1f} | {med(4):.1f} | {med(5):.1f} | {v[0][6]} |')
    for rt in ('small', 'small-pair', 'small-instant', 'big'):
        if (rt, 'base') in groups and (rt, 'new') in groups:
            b, n = groups[(rt, 'base')], groups[(rt, 'new')]
            rps = statistics.median(x[1] for x in n) / statistics.median(x[1] for x in b) - 1
            cpu = statistics.median(x[2] for x in n) / statistics.median(x[2] for x in b) - 1
            print(f'{rt}: reviews/s new vs base {rps*100:+.1f}%, process CPU {cpu*100:+.1f}%  base runs {[x[1] for x in b]} new {[x[1] for x in n]}')
