"""Reproducible full-book draft translation; never send formulas or identifiers.

Source: arXiv:2510.21890v3. Google Translate supplies the initial prose draft.
Run from any directory: python scripts/translate.py
Cached requests make interrupted runs resumable. No credentials are required.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import re
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'original' / 'tex'
DEST = ROOT / 'zh'
CACHE = ROOT / 'build' / 'translation-cache'
PAT = re.compile(r'ZXQ\s*(\d{5})\s*QXZ', re.I)
MATH_ENVS = set('equation equation* align align* alignat alignat* gather gather* multline multline* eqnarray eqnarray* displaymath math flalign flalign* tikzpicture pspicture verbatim lstlisting'.split())
FULL_ARGS = set('label ref eqref pageref autoref nameref Cref cref crefrange Crefrange cpageref cite citep citet parencite textcite nocite url path include input includegraphics graphicspath bibliography bibliographystyle addbibresource vspace hspace vspace* hspace* fontsize setstretch setcounter addtocounter setlength addtolength color cellcolor rowcolor arrayrulecolor pagecolor thispagestyle pagestyle markright cline cmidrule rule raisebox rotatebox scalebox reflectbox phantom hphantom vphantom texttt'.split())
FULL_ARGS.update(['parbox', 'resizebox', 'fcolorbox', 'definecolor', 'setkeys', 'setlist'])
FULL_ARGS.update(['multicolumn', 'multirow'])
SECOND_LABEL = set('defnr problems msg problemr asumr thm thmp lem lemp corp proppnp proppp msgu'.split())
FIRST_ONLY = set('href textcolor colorbox fcolorbox'.split())
DEFINITIONS = set('newcommand renewcommand providecommand DeclareRobustCommand NewDocumentCommand RenewDocumentCommand newenvironment renewenvironment newtcolorbox newtcbtheorem newtheorem definecolor tikzset pgfplotsset'.split())
TEX_FILES = [
    'Preface.tex', 'notations.tex', 'Prologue-Deep-Generative-Modeling.tex',
    'Chapter-0-Diffusion-High-Level.tex',
    'Chapter-1-Varational-Perspective-From-VAE-to-DDPM.tex',
    'Chapter-2-Score-Based-Perspective-From-EBMs-to-NCSN.tex',
    'Chapter-3-Diffusion-Models-Today-Score-SDE-Framework.tex',
    'Chapter-4-Flow-Based-Perspective-From-NFs-to-Flow-Matching.tex',
    'Chapter-5-They-Are-All-Equivalent.tex',
    'Chapter-4-plus-OT-and-Diffusion-Model.tex',
    'Chapter-6-Guidance.tex',
    'Chapter-7-Sophisticated-Solvers-for-Fast-Sampling.tex',
    'Chapter-8-Post-Training-Methods-for-Fast-Sampling.tex',
    'Chapter-9-Learning-Fast-Generator-from-Scratch.tex',
    'Chapter-10-Epilogue-and-Outlook.tex',
    'A_Crash-Course-of-DE.tex', 'B_Continuity-Equation.tex',
    'C_SDE-and-Ito.tex', 'D_Proofs.tex',
]


def strip_comments(s):
    return re.sub(r'(?m)(?<!\\)%[^\n]*', '', s)


def group_end(s, start, op='{', cl='}'):
    if start >= len(s) or s[start] != op:
        return start
    level = 0
    j = start
    while j < len(s):
        if s[j] == '\\':
            j += 2
            continue
        if s[j] == op:
            level += 1
        elif s[j] == cl:
            level -= 1
            if level == 0:
                return j + 1
        j += 1
    raise ValueError('Unclosed group: ' + s[start:start + 100])


def next_group(s, start, op='{', cl='}'):
    k = start
    while k < len(s) and s[k].isspace():
        k += 1
    return (k, group_end(s, k, op, cl)) if k < len(s) and s[k] == op else (k, k)


def protect(s):
    """Return prose with opaque tokens, plus the exact protected source slices."""
    ranges = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == '$':
            delim = '$$' if s.startswith('$$', i) else '$'
            j = i + len(delim)
            while True:
                j = s.find(delim, j)
                if j < 0:
                    raise ValueError('Unclosed dollar math near ' + s[max(0, i-50):i+100])
                if j == 0 or s[j - 1] != '\\':
                    break
                j += 1
            j += len(delim)
            ranges.append((i, j)); i = j; continue
        if c == '\\':
            m = re.match(r'\\([A-Za-z@]+\*?|.)', s[i:], re.S)
            if not m:
                i += 1; continue
            cmd = m[1]
            j = i + m.end()
            if cmd in ('[', '('):
                close = '\\]' if cmd == '[' else '\\)'
                k = s.find(close, j)
                if k < 0:
                    raise ValueError('Unclosed math delimiter')
                j = k + len(close)
            elif cmd in ('begin', 'end'):
                a, b = next_group(s, j)
                env = s[a + 1:b - 1] if b > a else ''
                j = b
                if cmd == 'begin' and env in MATH_ENVS:
                    close = '\\end{' + env + '}'
                    k = s.find(close, j)
                    if k < 0:
                        raise ValueError('Unclosed environment ' + env)
                    j = k + len(close)
                else:
                    a, b = next_group(s, j, '[', ']')
                    if b > a:
                        j = b
                    count = {'tabular': 1, 'tabular*': 2, 'tabularx': 2,
                             'minipage': 1, 'adjustbox': 1}.get(env, 0) if cmd == 'begin' else 0
                    for _ in range(count):
                        a, b = next_group(s, j)
                        if b > a: j = b
                    if cmd == 'begin' and env in ('mydefinition', 'myproblem', 'myresearchproblem', 'myassumption', 'mytheorem', 'mylemma', 'mycorollary', 'myproposition', 'myclaim', 'myfact'):
                        a, b = next_group(s, j)
                        a2, b2 = next_group(s, b)
                        if b2 > a2: ranges.append((a2, b2))
            elif cmd in DEFINITIONS:
                # Definitions contain parameter syntax, styles, and program text.
                # They are kept verbatim and their visible labels localized separately.
                k = j
                for _ in range(8):
                    a, b = next_group(s, k, '[', ']')
                    if b > a: k = b; continue
                    a, b = next_group(s, k)
                    if b > a:
                        k = b
                        if cmd in ('newcommand', 'renewcommand', 'providecommand', 'DeclareRobustCommand'):
                            a2, b2 = next_group(s, k, '[', ']')
                            if b2 > a2:
                                k = b2
                                a2, b2 = next_group(s, k, '[', ']')
                                if b2 > a2: k = b2
                            a2, b2 = next_group(s, k)
                            if b2 > a2: k = b2
                            break
                        if cmd in ('NewDocumentCommand', 'RenewDocumentCommand'):
                            for _ in range(2):
                                a2, b2 = next_group(s, k)
                                if b2 > a2: k = b2
                            break
                        if cmd == 'definecolor':
                            for _ in range(2):
                                a2, b2 = next_group(s, k)
                                if b2 > a2: k = b2
                            break
                        break
                    break
                j = k
            else:
                while True:
                    a, b = next_group(s, j, '[', ']')
                    if b <= a: break
                    j = b
                if cmd == 'cmidrule':
                    a, b = next_group(s, j, '(', ')')
                    if b > a: j = b
                if cmd in FULL_ARGS or cmd in FIRST_ONLY:
                    a, b = next_group(s, j)
                    if b > a: j = b
                if cmd in ('setlength', 'addtolength', 'setcounter', 'addtocounter', 'fontsize', 'rule', 'resizebox', 'fcolorbox', 'setkeys', 'multicolumn', 'multirow'):
                    a, b = next_group(s, j)
                    if b > a: j = b
                if cmd in SECOND_LABEL:
                    a, b = next_group(s, j)
                    a2, b2 = next_group(s, b)
                    if b2 > a2: ranges.append((a2, b2))
            ranges.append((i, j)); i = j; continue
        if c in '{}&~#_^':
            ranges.append((i, i + 1))
        i += 1
    # Adjacent protected pieces need only one marker.
    merged = []
    for a, b in sorted(ranges):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    out, pieces, k = [], [], 0
    for a, b in merged:
        out.append(s[k:a])
        out.append(f'ZXQ{len(pieces):05d}QXZ')
        pieces.append(s[a:b])
        k = b
    out.append(s[k:])
    return ''.join(out), pieces


def split_chunks(s, size=3000):
    chunks = []
    while len(s) > size:
        pos = s.rfind('\n\n', 0, size)
        if pos < size // 3:
            pos = s.rfind('\n', 0, size)
        if pos < size // 3:
            pos = s.rfind(' ', 0, size)
        if pos < 1: pos = size
        # A split must never bisect a protected token.
        for m in PAT.finditer(s[max(0, pos-18):pos+18]):
            a = max(0, pos-18) + m.start()
            b = max(0, pos-18) + m.end()
            if a < pos < b: pos = a
        chunks.append(s[:pos]); s = s[pos:]
    if s: chunks.append(s)
    return chunks


rate_lock = threading.Lock()
next_time = 0.


def request_translation(s):
    global next_time
    key = hashlib.sha256(s.encode()).hexdigest()
    path = CACHE / (key + '.json')
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))['translation']
    last = None
    for attempt in range(5):
        try:
            with rate_lock:
                delay = max(0, next_time - time.monotonic())
                next_time = max(next_time, time.monotonic()) + .28
            if delay: time.sleep(delay)
            url = 'https://translate.googleapis.com/translate_a/single?' + urllib.parse.urlencode({
                'client': 'gtx', 'sl': 'en', 'tl': 'zh-CN', 'dt': 't', 'q': s,
            })
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=50) as r:
                data = json.load(r)
            result = ''.join(x[0] for x in data[0] if x[0])
            result = PAT.sub(lambda m: f'ZXQ{int(m[1]):05d}QXZ', result)
            path.write_text(json.dumps({'source': s, 'translation': result}, ensure_ascii=False), encoding='utf-8')
            return result
        except Exception as e:
            last = e
            time.sleep(1 + attempt * 2)
    raise RuntimeError(f'Translation failed: {last}')


def english_prose(s):
    return bool(re.search(r'[A-Za-z]{2,}', PAT.sub('', s)))


def translate_chunk(s):
    if not english_prose(s): return s
    out = request_translation(s)
    expected = PAT.findall(s)
    if PAT.findall(out) == expected:
        return out
    # Some translations reorder tokens. Keep the source skeleton fixed and
    # translate the individual text leaves in that exceptional chunk.
    parts, last = [], 0
    leaves = []
    for m in PAT.finditer(s):
        raw = s[last:m.start()]
        leaves.append(raw)
        parts.append(len(leaves) - 1)
        parts.append(m[0]); last = m.end()
    leaves.append(s[last:])
    parts.append(len(leaves) - 1)
    translated = translate_leaves(leaves)
    return ''.join(translated[x] if isinstance(x, int) else x for x in parts)


def translate_leaves(leaves):
    selected = [(i, x) for i, x in enumerate(leaves) if re.search(r'[A-Za-z]{2,}', x)]
    result = list(leaves)
    if not selected: return result
    batch = '\n'.join(f'QX{i:05d}XQ: ' + re.sub(r'\s+', ' ', x.strip()) for i, x in selected)
    if len(batch) > 4500:
        half = len(leaves) // 2
        return translate_leaves(leaves[:half]) + translate_leaves(leaves[half:])
    out = request_translation(batch)
    records = list(re.finditer(r'QX\s*(\d{5})\s*XQ\s*[:：]?', out, re.I))
    got = {}
    for k, m in enumerate(records):
        end = records[k+1].start() if k+1 < len(records) else len(out)
        got[int(m[1])] = out[m.end():end].strip()
    for i, raw in selected:
        if i in got and got[i]:
            a = len(raw) - len(raw.lstrip())
            b = len(raw.rstrip())
            result[i] = raw[:a] + got[i] + raw[b:]
        else:
            result[i] = translate_leaf(raw)
    return result


def translate_leaf(s):
    if not re.search(r'[A-Za-z]{2,}', s): return s
    a = len(s) - len(s.lstrip())
    b = len(s.rstrip())
    out = request_translation(s[a:b])
    return s[:a] + out + s[b:]


def terminology(s):
    # Small, explicit terminology correction table; no changes inside math.
    replacements = {
        '培训': '训练', '流量匹配': '流匹配', '标准化流量': '归一化流',
        '噪声条件评分网络': '噪声条件分数网络', '基于可能性': '基于似然',
        '标准化密度': '归一化密度', '变量变化公式': '变量替换公式',
        '洁净样品': '干净样本', '嘈杂的观察': '含噪观测', '嘈杂的观测': '含噪观测',
        '埃尔博': 'ELBO', '最大可能性': '最大似然', '对数可能性': '对数似然',
        '潜在变量': '潜变量', '变分自动编码器': '变分自编码器',
        '得分匹配': '分数匹配', '得分函数': '分数函数', '得分模型': '分数模型',
        '基于得分': '基于分数', '去噪得分': '去噪分数', '得分估计': '分数估计',
        '正向损坏': '前向扰动', '前向损坏': '前向扰动', '正向腐蚀': '前向扰动',
        '前向腐败': '前向扰动', '正向腐败': '前向扰动', '腐败过程': '扰动过程',
        '损坏过程': '扰动过程', '腐化过程': '扰动过程',
        '标准化流': '归一化流', '归一化流程': '归一化流',
        '分类器免费指导': '无分类器引导', '无分类器指导': '无分类器引导',
        '分类器指导': '分类器引导', '概率流常微分方程': '概率流常微分方程',
        '流匹配': '流匹配', '朗之万': '朗之万',
    }
    for a, b in replacements.items(): s = s.replace(a, b)
    return s


def prepare():
    DEST.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for f in SOURCE.iterdir():
        if f.is_file() and f.name not in TEX_FILES:
            if not (DEST / f.name).exists(): shutil.copy2(f, DEST / f.name)
    if not (DEST / 'Images').exists(): shutil.copytree(SOURCE / 'Images', DEST / 'Images')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    prepare()
    jobs, docs = [], {}
    for name in TEX_FILES:
        source = strip_comments((SOURCE / name).read_text(encoding='utf-8'))
        masked, pieces = protect(source)
        chunks = split_chunks(masked)
        docs[name] = {'source': source, 'masked': masked, 'pieces': pieces,
                      'chunks': chunks, 'translated': [None] * len(chunks)}
        for i, chunk in enumerate(chunks): jobs.append((name, i, chunk))
    print(f'{len(docs)} files; {len(jobs)} chunks; ' +
          f'{sum(len(x["source"]) for x in docs.values())} active source characters', flush=True)
    done = 0
    failures = []
    started = time.monotonic()
    with cf.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(translate_chunk, s): (name, i) for name, i, s in jobs}
        for f in cf.as_completed(futures):
            name, i = futures[f]
            try:
                docs[name]['translated'][i] = f.result()
                done += 1
                if done % 10 == 0: print(f'{done}/{len(jobs)} chunks, {time.monotonic()-started:.0f}s', flush=True)
            except Exception as e:
                failures.append({'file': name, 'chunk': i, 'error': str(e)})
                print('FAILED', name, i, str(e), flush=True)
            doc = docs[name]
            if all(x is not None for x in doc['translated']):
                translated = terminology(''.join(doc['translated']))
                if PAT.findall(translated) != PAT.findall(doc['masked']):
                    raise ValueError('Protected token mismatch: ' + name)
                restored = PAT.sub(lambda m: doc['pieces'][int(m[1])], translated)
                (DEST / name).write_text(restored, encoding='utf-8')
                print('SAVED', name, flush=True)
    if failures:
        (ROOT/'build/translation-errors.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
        raise RuntimeError(f'{len(failures)} chunks failed; rerun to resume')
    report = {'source': 'arXiv:2510.21890v3', 'translator': 'Google Translate (gtx), initial draft',
              'chunks': len(jobs), 'active_source_characters': sum(len(x['source']) for x in docs.values()),
              'files': [{'file': name, 'protected_pieces': len(d['pieces']), 'chunks': len(d['chunks']),
                         'protected_sequence_preserved': True,
                         'chinese_characters': len(re.findall('[\u4e00-\u9fff]', (DEST/name).read_text(encoding='utf-8')))}
                        for name, d in docs.items()]}
    (ROOT/'translation-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('TRANSLATION COMPLETE', flush=True)


if __name__ == '__main__': main()
