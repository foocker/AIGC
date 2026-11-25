"""Verify chapter coverage, immutable identifiers, PDF text, and build warnings."""
from pathlib import Path
import hashlib
import json
import re
import sys
import pymupdf
from translate import ROOT, SOURCE, DEST, TEX_FILES, strip_comments


def tokens(s, command):
    return re.findall(r'\\'+command+r'\*?(?:\[[^\]]*\])*\{([^}]+)\}',s)


def math_blocks(s):
    from finalize import MATH_BLOCK

    visible = re.sub(r'(?m)(?<!\\)%[^\n]*', lambda m: ' ' * len(m[0]), s)
    excluded = ('tikzpicture', 'pspicture', 'verbatim', 'lstlisting')
    named_blocks = list(MATH_BLOCK.finditer(visible))
    # Mask named environments before finding bracketed displays, so a line
    # break with optional spacing inside a named block cannot be a match.
    outside = list(visible)
    for match in named_blocks:
        outside[match.start():match.end()] = ' ' * len(match[0])
    bracketed = re.finditer(r'(?<!\\)\\\[.*?(?<!\\)\\\]', ''.join(outside), re.S)
    displays = sorted(
        [match for match in named_blocks if match[1] not in excluded] + list(bracketed),
        key=lambda match: match.start(),
    )
    out = []
    for match in displays:
        block = strip_comments(s[match.start():match.end()])
        # The translated reference and introductory prose leave the formula intact.
        block = re.sub(r'\\(?:mathrm|text)\{(\\Cref\{[^}]+\})\}', r'\1', block)
        block = re.sub(r'\\text\{\s*(?:where|其中)\s*\}', r'\\text{where}', block)
        out.append(re.sub(r'\s+', '', block))
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    results=[]
    for name in TEX_FILES:
        a=(SOURCE/name).read_text(encoding='utf-8')
        b=(DEST/name).read_text(encoding='utf-8')
        if name=='notations.tex':
            results.append({'file':name,'manually_edited_notation_explanations':True})
            continue
        a=strip_comments(a);b=strip_comments(b)
        ids={}
        for key,cmd in [('labels','label'),('references','(?:Cref|cref|ref|eqref|pageref)'),
                        ('citations','(?:cite|citep|citet|parencite|textcite)'),('images','includegraphics')]:
            aa=tokens(a,cmd);bb=tokens(b,cmd)
            ids[key]={'source':len(aa),'translation':len(bb),'same_sequence':aa==bb}
        aa=math_blocks((SOURCE/name).read_text(encoding='utf-8'))
        bb=math_blocks((DEST/name).read_text(encoding='utf-8'))
        if aa!=bb:raise ValueError('Equation differs: '+name)
        if not all(x['same_sequence'] for x in ids.values()):raise ValueError('Identifier differs: '+name)
        results.append({'file':name,'identifiers':ids,'display_math_blocks':len(aa),'display_math_preserved':True,
                        'chinese_characters':len(re.findall('[\u4e00-\u9fff]',b)),
                        'sha256':hashlib.sha256((DEST/name).read_bytes()).hexdigest()})
    pdf=ROOT/'principles-diffusion-models-zh.pdf'
    doc=pymupdf.open(pdf)
    texts=[p.get_text() for p in doc]
    alltext='\n'.join(texts)
    log=(ROOT/'build/latex/main_zh.log').read_text(encoding='utf-8',errors='replace')
    missing=re.findall(r'Missing character[^\n]*',log)
    unresolved=re.findall(r'[^\n]*(?:undefined on input|undefined references|Citation .+ undefined|multiply defined)[^\n]*',log)
    leaks=[x for x in ['ZXQ','QXZ','TODO:','本小节内容将在后续任务中添加'] if x in alltext]
    report={'source':'arXiv:2510.21890v3','date':'2026-11-25','translation_status':'Full-book translation with a terminology and selected-passage review; paragraph-by-paragraph academic review pending',
            'pdf':pdf.name,'pdf_pages':len(doc),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'original_pdf_pages':len(pymupdf.open(ROOT/'original/principles-diffusion-models-v3-en.pdf')),
            'chapter_and_appendix_bookmarks':[x for x in doc.get_toc() if x[0]==2],
            'missing_glyph_warnings':missing,'unresolved_reference_or_citation_warnings':unresolved,
            'placeholder_leaks':leaks,'overfull_hbox_warnings':len(re.findall(r'Overfull \\hbox',log)),
            'content_files':results,
            'editorial_changes':['Chapter/section terminology corrections','Manual abstract, acknowledgements and notation explanations','Chinese theorem and navigation labels','Chinese command termination spacing','Original comments restored inside math/TikZ environments','Wider text block for long formulas','One mathrm reference changed to text for Chinese glyphs'],
            'original_figures_and_bibliography_kept_in_original_language':True}
    (ROOT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['content_files','chapter_and_appendix_bookmarks']},ensure_ascii=False,indent=2))
    if missing or unresolved or leaks:raise ValueError('PDF verification failed; see verification.json')


if __name__=='__main__':main()
