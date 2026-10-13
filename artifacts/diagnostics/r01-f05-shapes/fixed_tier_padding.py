import json, collections, math, sys
d=json.load(open('artifacts/diagnostics/r01-f05-shapes/audit-1/result.json'))
R=d['records']
def caps(rec):  # default_q8 bucket layout -> {(stream,len):count}
    b=rec['buckets']['default_q8']
    if not b['admitted']: return None
    return {(s,l):c for s,(ls,cs) in enumerate(zip(b['split_lens'],b['split_counts'])) for l,c in zip(ls,cs)}
def run(tiers_n, pct):
    tiers=collections.defaultdict(list)
    for r in R:
        n=r['labels_shape'][0]
        t=next(x for x in tiers_n if n<=x)
        tiers[t].append(r)
    total_orig=total_fixed=0; eager=0; graphs=0; hits=0
    for t,rs in sorted(tiers.items()):
        cs=[(r,caps(r)) for r in rs]
        ok=[x for x in cs if x[1]]
        keys=set(k for _,c in ok for k in c)
        cap={}
        for k in keys:
            vals=sorted(c.get(k,0) for _,c in ok)
            cap[k]=vals[min(len(vals)-1,int(math.ceil(pct*len(vals)))-1)]
        fixed=sum(l*c for (s,l),c in cap.items())
        # final stream n is the tier capacity: fixed positions for user stream use t
        # updates that fit
        fit=[r for r,c in ok if all(c.get(k,0)<=cap[k] for k in keys)]
        graphs+=1
        hits+=max(0,len(fit)-1)
        eager+=len(rs)-len(fit)
        for r in fit:
            total_orig+=r['recurrent_positions']; total_fixed+=fixed
        print(' tier',t,'updates',len(rs),'fit',len(fit),'fixed positions',fixed,'mean actual',sum(r['recurrent_positions'] for r in fit)//max(1,len(fit)))
    print('tiers',tiers_n,'pct',pct,'graphs',graphs,'replays',hits,'eager',eager,'overhead %.1f%%'%(100*(total_fixed/total_orig-1)))
for tn in ([512],[128,512],[32,128,512],[16,64,256,512]):
    for pct in (1.0,0.9):
        run(tn,pct)
