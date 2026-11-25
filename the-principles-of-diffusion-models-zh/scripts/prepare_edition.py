"""Apply the Chinese edition's explicit typography and frontmatter changes."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'original/tex'
ZH = ROOT/'zh'

FRONT = r'''
\begin{abstract}
本书聚焦于推动扩散模型发展的基本原理，追溯这些思想的起源，并阐明不同表述如何源于共同的数学思想。

扩散建模首先指定一个\emph{前向扰动过程}，将数据逐渐转化为噪声。这个过程定义了一族连续的中间分布，从而把数据分布与简单的噪声分布联系起来。扩散模型的核心目标，是构造一个沿相反方向运行的过程：将噪声转化为数据，同时重现前向扰动过程所定义的同一族中间分布。

我们从三个互补的视角形式化这一思想。受变分自编码器启发的\emph{变分视角}，将扩散看作逐步学习去噪：通过一系列局部去噪目标，教会模型将噪声还原为数据。源于能量模型的\emph{分数视角}，学习演化中分布的对数密度梯度，以确定应当如何将样本推向概率更高的区域。与归一化流相关的\emph{流视角}，则将生成看作在学习得到的速度场作用下，沿平滑路径把噪声样本传输到数据分布。

这些视角共享同一核心结构：一个学习得到、随时间变化的速度场，其诱导的流将简单先验分布传输为数据分布。在这个框架下，采样就是求解微分方程，使噪声沿连续的生成轨迹演化为数据。在此基础上，本书讨论用于可控生成的\emph{引导方法}、用于高效采样的\emph{高阶数值求解器}，以及受扩散模型启发、直接学习轨迹上任意两个时刻之间映射的\emph{流映射模型}。

本书面向具备基本深度学习背景，希望从概念和数学层面清晰理解扩散模型的读者。书中阐述这一领域的核心原理，解释统一不同表述的思想，并为在这一迅速发展的领域继续学习奠定基础。它既可作为研究人员的原理参考，也可作为学生与初学者的入门读物。配套材料见原著网站：
{\centering\url{https://the-principles-of-diffusion-models.github.io/}\par}

\medskip
\noindent\textbf{译本说明。}本文件是基于 arXiv:2510.21890v3（2026年8月27日）的非官方中文初译，整理日期为2026年11月25日。正文采用机器初译，并统一部分术语；摘要、出版标识与通用界面文字另行整理。尚未逐段完成人工学术审校。公式、数学符号、交叉引用和文献标识保留原文；原始配图与参考文献条目保留原语。阅读涉及定理条件、证明细节和算法实现的内容时，可对照同目录的官方英文源码。
\end{abstract}

\begin{acknowledgements}
我们衷心感谢社区中提出宝贵反馈、帮助发现勘误的各位同仁。特别感谢以下人士，姓名按字母顺序排列：\emph{Francis Bach}、\emph{Ilia Badanin}、\emph{Rwiddhi Chakraborty}、\emph{Mauricio Delbracio}、\emph{Jacob Lessing}、\emph{Kaiming He}、\emph{Yutong (Kelly) He}、\emph{Durk Kingma}、\emph{Shucheng Li}、\emph{Ramtin Moslemi}、\emph{Rukmangadh Sai Myana}、\emph{Stefano Peluchetti}、\emph{Yuri Plotkin}、\emph{François Rozet}、\emph{Yair Shenfeld}、\emph{Molei Tao}、\emph{Mohamad Ternanni}和\emph{Baojian Zhou}，以及匿名审稿人与其他热心提供反馈的读者。

作者也特别感谢首尔市立大学与韩国高等科学院（KIAS）的\emph{Dohyun Kwon}教授，感谢他慷慨投入时间和精力参与本书的讨论。他仔细审阅了\Cref{ch:ot-eot}的部分内容，帮助确保陈述与证明的正确性，并通过多次富有价值的讨论澄清了表述。他不仅提出技术建议，还提供了深思熟虑的反馈，乐于分享自己的观点，在全书写作过程中给予我们持续的鼓励。我们衷心感谢他的支持与同事情谊，这些帮助使本书的最终版本更加完善。
\end{acknowledgements}
'''

LOCALIZATION = {
    'After this chapter, you will be able to...': '学完本章，你将能够：',
    'Reader route.': '阅读路径。', 'Algorithm': '算法',
    'Input:': '输入：', 'Output:': '输出：',
    'Affiliation reflects the institution at the time of the work.': '单位信息为开展本工作时的所属机构。',
}
SUPPORT = {
    'Central question.': '核心问题。', 'Mental model.': '直观图景。',
    'What is learned?': '学习什么？', 'Chapter path.': '本章路线。',
    'Why this section?': '本节的作用。', 'What this equation says.': '公式的含义。',
    'Takeaway.': '要点。', 'Proof idea.': '证明思路。',
    'Full details:': '完整细节：', 'Running Example:': '贯穿示例：',
    'Method Lens:': '方法解析：', 'Object.': '对象。', 'Signal.': '学习信号。',
    'Generation.': '生成机制。', 'Trade-off.': '优势与局限。',
    'Question answered.': '已回答的问题。', 'Learned object.': '学习的对象。',
    'Tractable signal.': '可计算的信号。', 'How generation works.': '生成过程。',
    'Bridge.': '下一步。', 'Optional route.': '选读路径。', 'Keep in mind:': '请记住：',
    'Proof for Theorem.': '定理的证明。', 'Proof for Lemma.': '引理的证明。',
    'Proof for Corollary.': '推论的证明。', 'Proof for Proposition.': '命题的证明。',
    'Proof for Claim.': '断言的证明。', 'Proof.': '证明。', 'Remark.': '注。',
    'Example:': '例：',
    '{Definition}': '{定义}', '{Problem Setup}': '{问题设定}',
    '{Research Problem}': '{研究问题}', '{Assumption}': '{假设}',
    '{Theorem}': '{定理}', '{Lemma}': '{引理}', '{Corollary}': '{推论}',
    '{Proposition}': '{命题}', '{Claim}': '{断言}', '{Fact}': '{事实}',
}
DIAGRAMS = {
    'Variational \\\\Autoencoder': '变分\\\\自编码器',
    'Energy-Based Model': '能量模型', 'Normalizing Flows': '归一化流',
    'Overview of Deep Generative Modeling': '深度生成模型概览',
    'Variational View': '变分视角', 'Score-Based View': '分数视角', 'Flow-Based View': '流视角',
    'Denoising Diffusion\\\\Probabilistic Model': '去噪扩散\\\\概率模型',
    'Noise Conditional\\\\Score Network': '噪声条件\\\\分数网络',
    'Gaussian\\\\Flow Matching': '高斯\\\\流匹配',
    'Continuous-Time Formulation': '连续时间表述', 'e.g., Score SDE': '例如分数SDE',
    'Unifying Principles': '统一原理', 'Conditional Strategy': '条件化策略',
    'Fokker-Planck Equation': 'Fokker--Planck方程',
    'Perspective': '视角', 'Origin': '起源', 'Diffusion Model': '扩散模型',
    'Generation with Diffusion Model': '使用扩散模型生成',
    'Steering Generation': '控制生成方向', 'Guidance': '引导',
    'Fast Generation': '快速生成', 'with Numerical Solvers': '通过数值求解器',
    'Estimating the Integration': '近似计算积分', 'Learning a Fast': '学习快速的',
    'Diffusion-Based Generator': '扩散生成器', 'Learning the Integration': '学习积分映射',
    'Solve the ODE backward from': '反向求解常微分方程，从',
    ' to ': ' 到 ', ' with ': '，初值为 ', '(more generally, from': '（更一般地，从',
    '{Chapter}': '{章}', 'Chapter~': '章~', ' and ': '和',
}


def main():
    s = (SRC/'main_v2.tex').read_text(encoding='utf-8')
    s = re.sub(r'(?m)(?<!\\)%[^\n]*', '', s)
    # The source's publisher badge belongs to the English original, not this draft.
    s = s[:s.index('\\usepackage{eso-pic}')] + s[s.index('\\begin{document}'):]
    s = re.sub(r'\\usepackage\[utf8\]\{inputenc\}\s*', '', s)
    s = s.replace('\\usepackage{kotex}', '')
    s = s.replace('\\usepackage{pstricks}', '').replace('\\usepackage{pstricks-add}', '')
    s = s.replace('\\documentclass[examplefnt,biber]{nowfnt-arxiv}',
                  '\\makeatletter\\def\\input@path{{zh/}}\\makeatother\n'
                  '\\documentclass[examplefnt,biber]{nowfnt-arxiv}\n'
                  '\\usepackage[UTF8,scheme=plain,fontset=windows]{ctex}\n'
                  '\\setmainfont{TeX Gyre Termes}\n'
                  '\\setsansfont{TeX Gyre Heros}\n'
                  '\\setmonofont{Latin Modern Mono}\n')
    s = s.replace('\\usepackage{graphicx}', '\\usepackage{graphicx}\n\\graphicspath{{zh/}}')
    s = s.replace('total={136mm,206mm}', 'total={144mm,206mm}')
    s = s.replace('headheight=14.4pt', 'headheight=29pt')
    s = s.replace('\\title{The Principles of Diffusion Models}',
                  '\\title{扩散模型原理}\n\\subtitle{非官方中文初译 · arXiv v3}')
    s = s.replace('\\addbibresource{main_v2.bib}', '\\addbibresource{zh/main_v2.bib}')
    s = s.replace('\\articledatabox{\\nowfntstandardcitation}',
                  '\\articledatabox{原著：arXiv:2510.21890v3；非官方中文初译；2026年11月25日。}')
    # Translate the front matter explicitly rather than passing publishing metadata
    # or author addresses through a translation service.
    a = s.index('\\begin{abstract}')
    b = s.index('\\end{acknowledgements}', a) + len('\\end{acknowledgements}')
    s = s[:a] + FRONT + s[b:]
    parts = {
        'Introduction to Deep Generative Modeling': '深度生成模型导论',
        'Foundations and Perspectives of Diffusion Models': '扩散模型的基础与视角',
        'Sampling of Diffusion Models': '扩散模型的采样',
        'Toward Learning Fast Diffusion-Based Generators': '学习快速的扩散生成器',
        'Epilogue and Outlook': '结语与展望',
    }
    for a, b in (LOCALIZATION | parts).items(): s = s.replace(a, b)
    custom = r'''
\renewcommand{\abstractname}{摘要}
\renewcommand{\acknowledgementsname}{致谢}
\renewcommand{\contentsname}{目录}
\renewcommand{\bibname}{参考文献}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\proofname}{证明}
\addto\captionsenglish{\renewcommand{\contentsname}{目录}\renewcommand{\abstractname}{摘要}}
\DefineBibliographyStrings{english}{bibliography={参考文献}}
\hypersetup{pdftitle={扩散模型原理（中文初译）},pdfsubject={arXiv:2510.21890v3 的非官方中文初译}}
\setlength{\emergencystretch}{2em}
\crefname{algorithm}{算法}{算法}
\Crefname{algorithm}{算法}{算法}
\crefname{figure}{图}{图}
\Crefname{figure}{图}{图}
\crefname{table}{表}{表}
\Crefname{table}{表}{表}
\crefname{equation}{式}{式}
\Crefname{equation}{式}{式}
\crefname{chapter}{章}{章}
\Crefname{chapter}{章}{章}
\crefname{section}{节}{节}
\Crefname{section}{节}{节}
\crefname{appendix}{附录}{附录}
\Crefname{appendix}{附录}{附录}
\crefname{theorem}{定理}{定理}
\Crefname{theorem}{定理}{定理}
\crefname{lemma}{引理}{引理}
\Crefname{lemma}{引理}{引理}
\crefname{proposition}{命题}{命题}
\Crefname{proposition}{命题}{命题}
\crefname{definition}{定义}{定义}
\Crefname{definition}{定义}{定义}
'''
    s = s.replace('\\begin{document}', custom + '\n\\begin{document}')
    (ROOT/'main_zh.tex').write_text(s, encoding='utf-8')
    # Localize the reusable theorem/proof/navigation chrome without changing
    # command names, counters, signatures, colors, or citation identifiers.
    for name in ['theorems.tex', 'scaffolding.tex']:
        t = (SRC/name).read_text(encoding='utf-8')
        for a, b in SUPPORT.items():
            if a == '{Theorem}':
                t = t.replace('}{Theorem}', '}{定理}')
            else: t = t.replace(a, b)
        (ZH/name).write_text(t, encoding='utf-8')
    for name in ['Part-B-diagram.tex', 'Part-C-D-diagram.tex']:
        t = (SRC/name).read_text(encoding='utf-8')
        for a, b in sorted(DIAGRAMS.items(), key=lambda x: -len(x[0])): t = t.replace(a, b)
        (ZH/name).write_text(t, encoding='utf-8')
    t = (SRC/'nowfnt-arxiv.cls').read_text(encoding='utf-8')
    t = t.replace('ABSTRACT\\par', '摘要\\par')
    (ZH/'nowfnt-arxiv.cls').write_text(t, encoding='utf-8')
    t = (SRC/'nowfnt-biblatex.sty').read_text(encoding='utf-8')
    t = t.replace('[References]', '[参考文献]')
    (ZH/'nowfnt-biblatex.sty').write_text(t, encoding='utf-8')
    t = (SRC/'math_commands.tex').read_text(encoding='utf-8')
    for a, b in [('(Left)', '（左）'), ('(Center)', '（中）'), ('(Right)', '（右）'),
                 ('(Top)', '（上）'), ('(Bottom)', '（下）')]: t = t.replace(a, b)
    for en, zh in [('figures', '图'), ('figure', '图'), ('Figure', '图'),
                   ('sections', '节'), ('section', '节'), ('Section', '节'),
                   ('equation', '式'), ('Equation', '式'), ('chapters', '章'),
                   ('chapter', '章'), ('Chapter', '章'), ('algorithms', '算法'),
                   ('algorithm', '算法'), ('Algorithm', '算法'), ('parts', '篇'),
                   ('part', '篇'), ('Part', '篇')]:
        t = re.sub(r'(?<=[{ ])' + en + r'(?=[ ~])', zh, t)
    (ZH/'math_commands.tex').write_text(t, encoding='utf-8')


if __name__ == '__main__': main()
