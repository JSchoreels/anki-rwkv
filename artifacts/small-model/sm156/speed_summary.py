import sys,re,collections,statistics as st
cur=None; rows=collections.defaultdict(list); loads=collections.defaultdict(list)
for line in open(sys.argv[1]):
    m=re.match(r'=== (\S+ \S+) cards=(\d+) rounds=(\d+) rep=(\d) binary=(\w+) precision=(\w+)',line)
    if m: cur=m.groups(); continue
    m=re.match(r'cpu_load_percent=(\d+)',line)
    if m: loads[cur[1]].append(int(m.group(1)))
    m=re.match(r'(big|small|tiny): .*reviews_per_s=(\d+)',line)
    if m and cur: rows[(cur[1],m.group(1),cur[4],cur[5])].append(int(m.group(2)))
short={'compact_cards_int8':'int8','compact_lowrank_int8':'T24'}
for shape in sorted(set(k[0] for k in rows),reverse=True):
  print('shape',shape,'load',min(loads[shape]),'-',max(loads[shape]),'median',st.median(loads[shape]))
  for size in ['big','small','tiny']:
    d={}
    for b in ['old','new']:
      for p in short:
        v=rows.get((shape,size,b,p))
        if not v: continue
        d[(b,short[p])]=(max(v),st.median(v),v)
    if not d: continue
    for k,(mx,md,v) in d.items(): print(f'  {size} {k[0]} {k[1]}: best {mx} ({1e6/mx:.1f} us) median {md} ({1e6/md:.1f} us) {v}')
    for est,i in [('best',0),('median',1)]:
      i8=1e6/((d[('old','int8')][i]+0)) ; i8n=1e6/d[('new','int8')][i]
      o=1e6/d[('old','T24')][i]; n=1e6/d[('new','T24')][i]
      print(f'  {size} {est}: int8 old {i8:.1f} new {i8n:.1f} | T24 old {o:.1f} new {n:.1f} | extra old {o-i8:.1f} new {n-i8n:.1f} us')
