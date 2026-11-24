"""Check preserved technical payloads, new reading order and untouched theory."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SNAPSHOT=ROOT/'build/seedance-style-before'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    spec=importlib.util.spec_from_file_location('editorial_record',ROOT/'scripts/rewrite_seedance_style.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts=dict(displays=0,environments=0,headings=0,urls=0,labels=0,codepaths=0,inline_math=0)
    for old_path in SNAPSHOT.glob('*.tex'):
        path=ROOT/'chapters/seedance'/old_path.name
        if not path.exists() or path.name=='contents.tex':
            continue
        old=old_path.read_text(encoding='utf-8')
        new=path.read_text(encoding='utf-8')
        for index,(a,b) in enumerate(zip(module.protected(old),module.protected(new))):
            if index==2:
                a=[re.sub(r'(\\chapter\{)M\d\d ',r'\1',x) for x in a]
            missing=Counter(a)-Counter(b)
            assert not missing,(path.name,index,list(missing.elements())[:2])
            counts[list(counts)[index]]+=len(a)
        inline=lambda s:re.findall(r'\\\(.*?\\\)|(?<!\\)\$(?!\$).*?(?<!\\)\$',s,re.S)
        missing=Counter(inline(old))-Counter(inline(new))
        assert not missing,(path.name,'inline math',list(missing.elements()))
        counts['inline_math']+=len(inline(old))
    baseline=json.loads((SNAPSHOT/'hashes.json').read_text(encoding='utf-8'))
    unchanged={n:h for n,h in baseline.items() if n.startswith('chapters/theory/') or
               n in ['AIGCTheory.tex','AIGCTheory.pdf','chapters/operations-and-applications.tex']}
    changed=[n for n,h in unchanged.items() if sha(ROOT/n)!=h]
    assert not changed,changed
    contents=(ROOT/'chapters/seedance/contents.tex').read_text(encoding='utf-8')
    inputs=re.findall(r'\\input\{chapters/seedance/([^}]+)\}',contents)
    assert inputs[0]=='architecture'
    assert len(inputs)==len(set(inputs))
    assert set(f'm{i:02d}' for i in range(28)).issubset(inputs)
    assert inputs.index('m01')<inputs.index('m03')<inputs.index('m02')<inputs.index('m11')
    source='\n'.join((ROOT/'chapters/seedance'/(n+'.tex')).read_text(encoding='utf-8') for n in inputs)
    labels=re.findall(r'\\label\{([^}]+)\}',source)
    labels+=['src-'+n for n in re.findall(r'\\sourceentry\{([^}]+)\}',source)]
    references=re.findall(r'\\(?:hyperref\[|(?:eqref|ref)\{)([^]}]+)',source)
    missing_refs=sorted(set(references)-set(labels))
    assert not missing_refs,missing_refs
    assert len(labels)==len(set(labels)),'Duplicate label'
    assert not any(ord(c)<32 and c not in '\n\t' for c in source)
    record={'date':'2026-11-24','version':'0.6','status':'passed',
            'protected_payload_counts':counts,
            'theory_and_operations_files_unchanged':len(unchanged),
            'new_reading_order':inputs,'missing_internal_references':missing_refs,
            'labels':len(labels),'references':len(references),
            'chapter_title_change':'Original title text retained; source Mxx prefix moved to source note.',
            'scope':'Source integrity and navigation only; not a model experiment.'}
    out=ROOT/'research/seedance-style/source-checks.json'
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':
    main()
