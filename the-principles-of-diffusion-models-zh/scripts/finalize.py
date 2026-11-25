"""Check the translated skeleton, restore comment-sensitive math, polish headings."""
import hashlib
import json
import re
from pathlib import Path
from translate import ROOT, SOURCE, DEST, TEX_FILES, protect, strip_comments

CHAPTERS = {
 'Prologue-Deep-Generative-Modeling.tex': '深度生成模型',
 'Chapter-1-Varational-Perspective-From-VAE-to-DDPM.tex': '变分视角：从VAE到DDPM',
 'Chapter-2-Score-Based-Perspective-From-EBMs-to-NCSN.tex': '分数视角：从能量模型到NCSN',
 'Chapter-3-Diffusion-Models-Today-Score-SDE-Framework.tex': '当代扩散模型：分数SDE框架',
 'Chapter-4-Flow-Based-Perspective-From-NFs-to-Flow-Matching.tex': '流视角：从归一化流到流匹配',
 'Chapter-5-They-Are-All-Equivalent.tex': '扩散模型的统一与系统视角',
 'Chapter-4-plus-OT-and-Diffusion-Model.tex': '扩散模型与最优传输（选读）',
 'Chapter-6-Guidance.tex': '引导与可控生成',
 'Chapter-7-Sophisticated-Solvers-for-Fast-Sampling.tex': '扩散快速采样的数值求解器',
 'Chapter-8-Post-Training-Methods-for-Fast-Sampling.tex': '基于蒸馏的快速采样方法',
 'Chapter-9-Learning-Fast-Generator-from-Scratch.tex': '从头学习快速生成器',
 'Chapter-10-Epilogue-and-Outlook.tex': '结语与展望',
 'A_Crash-Course-of-DE.tex': '微分方程速成',
 'B_Continuity-Equation.tex': '密度演化：从变量变换到福克尔-普朗克方程',
 'C_SDE-and-Ito.tex': '扩散模型背后的数学：伊藤微积分与吉尔萨诺夫定理',
 'D_Proofs.tex': '补充材料与证明',
}
MATH_BLOCK = re.compile(r'\\begin\{(equation\*?|align\*?|alignat\*?|gather\*?|multline\*?|eqnarray\*?|flalign\*?|displaymath|math|tikzpicture|pspicture|verbatim|lstlisting)\}.*?\\end\{\1\}', re.S)


def true_strip(s):
    return re.sub(r'(?m)(?<!\\)%[^\n]*(?:\n|$)', '', s)


def main():
    report = []
    for name in TEX_FILES:
        raw = (SOURCE/name).read_text(encoding='utf-8')
        target = (DEST/name).read_text(encoding='utf-8')
        # Chapter titles have an explicit editorial translation, including
        # replacing the original TeX accent/linebreak spelling of Itô.
        chapterless = lambda s: re.sub(r'\\chapter\{[^}]+\}', r'\\chapter{}', s, count=1)
        _, source_pieces = protect(chapterless(strip_comments(raw)))
        _, target_pieces = protect(chapterless(strip_comments(target)))
        if source_pieces != target_pieces:
            # On a repeat run, the only allowed differences are restored
            # environment whitespace and explicitly localized heading prose.
            normalize = lambda x: re.sub(r'\s+', '', x)
            if normalize(''.join(source_pieces)) != normalize(''.join(target_pieces)):
                import difflib
                a=normalize(''.join(source_pieces)); b=normalize(''.join(target_pieces))
                delta=next(x for x in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if x[0]!='equal')
                _,i,j,k,l=delta
                raise ValueError(f'{name}: protected source differs at {a[max(0,i-60):j+60]!r} / {b[max(0,k-60):l+60]!r}')
        visible = re.sub(r'(?m)(?<!\\)%[^\n]*', lambda m: ' '*len(m[0]), raw)
        source_blocks = list(MATH_BLOCK.finditer(visible))
        target_blocks = list(MATH_BLOCK.finditer(target))
        if [x[1] for x in source_blocks] != [x[1] for x in target_blocks]:
            raise ValueError('Environment sequence mismatch: '+name)
        for a, b in reversed(list(zip(source_blocks, target_blocks))):
            target = target[:b.start()] + raw[a.start():a.end()] + target[b.end():]
        if name in CHAPTERS:
            target = re.sub(r'\\chapter\{[^}]+\}', lambda _: '\\chapter{'+CHAPTERS[name]+'}', target, count=1)
        if name == 'Preface.tex':
            target = target.replace('前言和路线图', '前言与阅读路线')
        if name == 'Chapter-0-Diffusion-High-Level.tex':
            target = target.replace('扩散模型训练和采样的第一个操作视图', '先从训练与采样流程理解扩散模型')
        # Glossary corrections remain outside every protected source region.
        masked, pieces = protect(target)
        fixes = {
            '于是，一家人': '因此，这族分布', '分布家庭': '分布族',
            '噪声条件评分网络': '噪声条件分数网络', '流量匹配': '流匹配',
            '标准化流量': '归一化流', '标准化流': '归一化流',
            '变分自动编码器': '变分自编码器', '基于能量的模型': '能量模型',
            '最大可能性': '最大似然', '对数可能性': '对数似然',
            '埃尔博': 'ELBO', '潜在变量': '潜变量', '培训': '训练',
            '损坏规则': '扰动规则', '累积损坏': '累积扰动', '得分': '分数',
            '无分类器指导': '无分类器引导', '分类器指导': '分类器引导',
            '流程图': '流映射', '流图': '流映射', '地图': '映射',
            '边际分布': '边缘分布', '边际化': '边缘化',
            '身份': '恒等式', '力矩动力学': '矩的演化', '力矩': '矩',
            '无需挥手': '无需依赖非严格的论证', '提炼': '蒸馏',
        }
        for a,b in fixes.items(): masked = masked.replace(a,b)
        from translate import PAT
        target = PAT.sub(lambda m: pieces[int(m[1])], masked)
        target = target.replace('它\\^o', 'It\\^o')
        # XeTeX assigns letter catcodes to CJK. A command must be terminated
        # before directly adjacent translated Chinese (notably algorithmicx).
        target = re.sub(r'(\\[A-Za-z@]+\*?)(?=[\u4e00-\u9fff])', r'\1 ', target)
        (DEST/name).write_text(target, encoding='utf-8')
        report.append({'file': name, 'protected_pieces': len(source_pieces),
                       'formula_and_tex_skeleton_preserved': True,
                       'comment_sensitive_environments_restored': len(source_blocks),
                       'chinese_characters':len(re.findall('[\u4e00-\u9fff]',target)),
                       'sha256':hashlib.sha256(target.encode()).hexdigest()})
    (ROOT/'source-integrity.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding='utf-8')


if __name__ == '__main__': main()
