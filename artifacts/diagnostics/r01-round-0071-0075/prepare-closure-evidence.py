"""Select scoped stopped-round evidence and its persistent source/build dependencies."""
import hashlib,json,re
from pathlib import Path

repo=Path.cwd()
# Reconcile ordinary prose spacing only in the new closure paragraphs/rows.
documents=['docs/GPU_OPTIMIZATIONS.md','docs/R01_HOST_PATH.md',
           'artifacts/diagnostics/r01-round-0071-0075/summary.md',
           'artifacts/retraining/gpu-retrain-r01/codex-round-0071-0073-summary.md']
words=r'accepted|Accepted|historical|after|before|completed|next|previous|through|iteration|cycle|cycles|recorded|took|versus|mean|lower|of|all|and|full|control|user|seed|block|count|budget|from|saved|SD|is|remains|with|exact|false|no|No|The|training'
literal={'concluding0073':'concluding 0073','WKV0072':'WKV 0072','GroupNorm0073':'GroupNorm 0073',
         'native WKV72':'native WKV 72','native GroupNorm48':'native GroupNorm 48',
         'preparation0':'preparation 0','3of10':'3 of 10','7unconsumed':'7 unconsumed',
         '10-count':'10-count','capture10passes':'capture 10 passes','mean+':'mean +','lower−':'lower −','lower+':'lower +'}
for name in documents:
    path=Path(name);lines=path.read_text(encoding='utf-8').splitlines()
    for i,line in enumerate(lines):
        change=(i in (2,4)) if name.startswith('docs/') else (i<25 or line.startswith('- Human') or line.startswith('- Claude') or line.startswith('- After explicit'))
        if name=='docs/GPU_OPTIMIZATIONS.md':change=change or line.startswith(('| F65 |','| F66 |','| F70 |','| Exact native WKV accepted'))
        if not change:continue
        for a,b in literal.items():line=line.replace(a,b)
        line=re.sub(r'\b('+words+r')(?=\d)',r'\1 ',line)
        line=re.sub(r'(?<=\d)(?=(?:reviews/s|reviews\b|waves\b|blocks\b|batches\b|passes\b|s\b))',' ',line)
        line=re.sub(r',(?=\d)',', ',line) if not line.startswith('|') else line
        # Numeric thousands separators are not prose punctuation.
        line=re.sub(r'(?<=\d), (?=\d{3}(?:\D|$))',',',line)
        lines[i]=line
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
roots=[Path('artifacts/retraining/gpu-retrain-r01/iteration-'+n) for n in ('0071','0072','0073')]
roots += [Path('artifacts/diagnostics/r01-round-0071-0075'),Path('artifacts/diagnostics/r01-native-toolchain')]
suffixes={'.json','.jsonl','.log','.md','.txt','.csv','.py','.cpp','.ps1','.diff','.ncu-repz','.gz'}
selected=set()
for root in roots:
    for path in root.rglob('*'):
        if path.is_file() and path.suffix in suffixes and path.name!='accepted0042-trace.json':
            selected.add(path)
selected.update(Path('artifacts/retraining/gpu-retrain-r01').glob('codex-round-0071-0073-*'))
selected.update(Path('artifacts/retraining/gpu-retrain-r01/plans').glob('iteration-007[123]-codex.json'))
selected.update(Path('artifacts/retraining/gpu-retrain-r01').glob('iteration-007[123]*console.log'))
source_archives=set();cache_keys=set();missing=[]
def walk(value):
    if isinstance(value,dict):
        for item in value.values():walk(item)
    elif isinstance(value,list):
        for item in value:walk(item)
    elif isinstance(value,str):
        normalized=value.replace('\\','/')
        if normalized.startswith('artifacts/sources/') and normalized.endswith('.zip'):
            source_archives.add(Path(normalized))
        if re.fullmatch('[0-9a-f]{64}',value) and (Path('artifacts/cache/r01-native')/value/'build.json').is_file():
            cache_keys.add(value)
for path in list(selected):
    if path.suffix=='.json' and path.stat().st_size<8*1024*1024:
        try:walk(json.loads(path.read_text(encoding='utf-8')))
        except (UnicodeError,json.JSONDecodeError):pass
for archive in source_archives:
    if not archive.is_file():missing.append(str(archive));continue
    selected.add(archive)
    selected.update(path for path in archive.parent.glob('*.json') if path.is_file())
for key in cache_keys:
    folder=Path('artifacts/cache/r01-native')/key
    selected.update(path for path in folder.iterdir() if path.is_file() and path.suffix in {'.dll','.json','.def','.lib','.log'})
if missing:raise RuntimeError('Missing preserved source archives: '+str(missing))
out=Path('artifacts/diagnostics/r01-round-0071-0075')
manifest={'scope':'0071–0073 full cycles and preparatory/toolchain/offline baseline evidence; checkpoints and uncompressed large Chrome trace remain locally preserved',
          'source_archives':[str(p).replace('\\','/') for p in sorted(source_archives)],'native_build_keys':sorted(cache_keys),
          'files':len(selected),'bytes':sum(p.stat().st_size for p in selected),
          'largest':[{'path':str(p).replace('\\','/'),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(selected,key=lambda p:p.stat().st_size,reverse=True)[:8]]}
(out/'closure-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
selected.add(out/'closure-evidence-manifest.json')
(out/'closure-evidence-paths.txt').write_text('\n'.join(str(p).replace('\\','/') for p in sorted(selected))+'\n',encoding='utf-8')
print(json.dumps({'files':len(selected),'bytes':manifest['bytes'],'source_archives':len(source_archives),'native_builds':len(cache_keys),'missing':missing}))
