# 原理篇：保留原稿、局部修订与原位增补

2026-11-24，v0.5，本地审阅稿。

## 本轮：补齐参考书中的理论缺口

v0.5 保持 v0.4 的四章正文与所有原位补充不变，新增第5--10章理论内容。新增部分沿用原稿“提出问题—给定假设—逐步推导—解释用途”的写法，不以综合提纲代替现有内容。

新增六章分别展开：潜变量/能量模型/连续流，流匹配与训练等价，最优传输/Schrödinger 桥/梯度流，条件引导/逆问题/偏好对齐，数值求解/蒸馏/流映射，以及离散扩散训练/采样/误差控制。具体缺口、参考书节号与新增位置见 [理论缺口对照](theory-gaps.md)。

新增源文件在 `chapters/theory/supplements/`，统一由 `chapters/theory/contents.tex` 接入。`restoration.json` 与 `local-corrections.json` 记录此前恢复原稿的工作；本轮并未再次重写那份正文。当前文件哈希核对与 PDF 检查记录在 `verification.json`。

新增公式另用 `python scripts/check_theory_supplements.py` 做数值代入与有限状态枚举核对，结果在 `supplement-math-checks.json`。这些检查帮助发现转录错误，不是一般定理证明或模型实验。

## 此前：恢复原稿与原位补充

v0.4 按作者要求恢复原稿。此前将理论部分改为13章综合讲解的版本已停用，成品保留“零—总体介绍—数学基础—扩散模型”四章，以及原有的章、节顺序、个人阅读笔记、公式展开、图示与来源链接。此前修正集中在局部错误和衔接问题，原位解释接在原有推导或空节处。

## 文件与编译

- `AIGCTheory.pdf` / `AIGCTheory.tex`：原理篇独立读本。
- `StabelDiffusion.pdf` / `StabelDiffusion.tex`：原理篇、简短部署与应用章节、既有 Seedance 专题。
- `chapters/theory/original-foundations.tex`：恢复的原稿及局部修订；保留长推导主体。
- `chapters/theory/additions/`：通过 `\input` 插入原有位置的补充，可逐项审阅。
- `research/theory-update/local-corrections.json`：逐项记录修改前后文本及原因。
- `research/theory-update/restoration.json`：基准提交、原稿行保留情况及标题顺序核对。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_books.ps1 -Target theory
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_books.ps1 -Target full
python scripts/check_theory_math.py
python scripts/check_theory_supplements.py
python scripts/review_books.py --books AIGCTheory StabelDiffusion
```

编译需要 TeX Live / XeLaTeX / latexmk / Biber，不需要参考书 PDF。数值代入检查使用 Python 标准库，PDF 渲染检查使用 PyMuPDF 与 Pillow。成品在仓库根目录，编译与渲染中间文件在忽略的 `build/`。

`scripts/restore_theory_locally.py` 是本次恢复的可复核记录，会从固定基准重新生成理论主体并覆盖同名文件；日常编译不运行它。继续人工修改书稿后，请勿直接重放该脚本覆盖新改动。

## 保留与新增的关系

| 原有位置 | 本版处理 |
|---|---|
| 符号说明、基本介绍 | 填空项，补清加噪、学习、采样三个问题；个人开篇保留 |
| 高斯、熵、KL、重参数 | 保留 ELBO 原推导，补密度、微分熵、重参数采样与高斯 KL 的联系 |
| 贝叶斯、Fisher/Stein、MCMC | 原有标题和链接保留，补高斯后验例子、求导变量、Stein 恒等式和 Langevin 不变密度 |
| 证据下界 | 保留两套逐行推导，说明 ELBO 与负 ELBO 的符号约定 |
| SDE、布朗运动、Itô、FPE | 保留定义、证明及例子；修正差商、概率界、四阶矩、漂移与条件期望等局部错误，补等距与连续性方程的过渡 |
| 离散版 DDPM | 保留闭式加噪、后验配方、均值化简及完整 ELBO 分解；补后验条件区别与噪声损失的精确权重 |
| NCSN / score matching | 原有叙述、公式、退火采样图保留；空节补分部积分、DSM 条件平均证明与 Tweedie 关系 |
| 极大似然、连续 Score SDE | 补 ODE 散度积分、概率流的二分之一系数、反向时钟和参数化换算，不以短概括替代原推导 |
| 一些理论结果 | 将原先错放在 ELBO 后的误差界移到此处，解释先验、分数与数值误差的作用及条件 |
| DDIM、DPM++、蒸馏 | 接在原文后补跳步公式、积分因子与一阶等价，再补教师轨迹与分布匹配两种蒸馏机制 |
| 一致性 / LCM | 原公式与图保留，修正边界笔误，补 iCT、CTM 与 MeanFlow 的衔接及平均速度恒等式 |
| 离散扩散 | 在连续模型与应用之间新增短节，解释反向跳率、SEDD 的概率比与掩码恢复概率 |
| 模块、Rectified Flow、条件控制 | 填原有空节，补 CFM 条件平均及连续性推导、reflow、OT/SB 区别与 CFG；原交叉注意力公式移回模块位置 |
| 其它应用与框架提纲 | 保留标题，简短补上概念关系；旧训练预算注明是历史笔记 |

完整的标题列表、逐行保留数量和比例见 `restoration.json`。该比例统计完全相同的非空源代码行，不能代替语义审阅；公式原步骤保留，排版与少量符号修正也计作行变化。

此前的13章精简稿保存在 `research/theory-update/superseded-condensed-draft/`，不参与编译；相关原创示意图保留为资料，当前书稿不靠这些图替换原推导。`examples/legacy-theory-20261124.tex` 仍保留旧稿快照。

## 参考与范围

主参考是作者指定的 [The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/)，核验版本为 arXiv:2510.21890v3，2026-08-27，下载 PDF 共522页。v3印刷页码比 PDF 页序少5；网页部分页码对应早期版本，本版按章节核对。

| 补充内容 | 主参考定位 |
|---|---|
| 变分、DDPM | 第1–2章 |
| 分数匹配、NCSN | 第3章 |
| SDE、FPE、概率流与似然 | 第4章及附录 A–C |
| CFM、参数化与运输区别 | 第5–7章 |
| 条件引导 | 第8章 |
| 数值采样与蒸馏 | 第9–10章 |
| 一致性、轨迹映射、MeanFlow | 第11章，MeanFlow 见11.5节 |
| 离散扩散 | 第12.2节 |

补充同时使用 `theory.bib` 登记的原论文固定版本。来源身份与下载记录见 `primary-sources.json`、`manifest.json`；不是声称逐页精读每篇论文或穷尽截至本日的全部前沿工作。网站的勘误 PDF 含占位示例，未作为已确认勘误使用。全文与网页快照留在忽略的 `build/theory-sources/`。

既有 Seedance 专题的源码、独立主文件与独立 PDF 本轮未改。用户此前要求处理的部署、商业应用章节继续保留简短补充，围绕 vLLM/SGLang 相关工具与设计、短视频、短剧工作流；恢复原稿后它们在完整书稿中的编号自然前移。

## 检查

本轮重新编译原理篇及完整书稿，检查缺字、引用、公式溢出、内部跳转并渲染页面，记录在 `verification.json`。重点放大检查原有长公式、新增推导与章节衔接。

`check_theory_math.py` 用固定输入核对高斯 Tweedie、DDPM 后验、参数换算、DPM/DDIM 一阶更新、OU 方差、MeanFlow 恒等式与掩码反向概率。它能帮助发现符号与转录问题，不构成一般定理证明，也不验证模型效果。本轮没有训练、GPU 推理或业务实测。
