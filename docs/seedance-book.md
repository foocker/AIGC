# Seedance 2.5 书稿：阅读与编译

2026-11-24，本地审阅稿，未推送远端。

## 先打开什么

`Seedance25.pdf` 是本次新增内容的独立读本，`StabelDiffusion.pdf` 是接入原有扩散模型内容后的完整书稿。两者共享 `chapters/seedance/`，修改同一份 TeX 即可同时更新。PDF 目录、模块之间的引用、来源登记与16份机制代码均可点击跳转；正文不需要访问网络才能阅读。

专题的数学导读新增了 VAE 的 ELBO 与高斯 KL、流匹配的条件均值推导、连续性方程、速度/噪声/干净样本/分数之间的换算、时空 token、双向音视频交换、条件接口图及注意力资源算例。它用于衔接原书的 DDPM、Score SDE 与后续模块。

| 阅读目标 | 章节入口 |
|---|---|
| 先看完整方案和主要投入 | M27，随后看 M19–M26 |
| 从数学与基础模块开始 | 数学导读，M01–M05 |
| 多参考、编辑和白模 | M07–M08、M16–M18、M21–M23 |
| 长叙事、因果续写与缓存 | M06、M19、M24 |
| 数据、训练和推理成本 | M09–M13、M20、M25–M26 |
| 真实工程与教学片段 | M15、机制代码、来源登记 |

保留了 M00–M27 的编号以便与原材料对照，章节标题作了缩短；跨文件链接改为书内跳转，其它工程链接指向固定版本上游。

## 内容与证据边界

原材料来自 `multimodal-journey` 的干净工作区，commit `0f1b6bb86d609d75d8c2582063d85707d0180d39`。AIGC 初始 commit 为 `0a5ee3d50e9fb6a6d1d3f0a44c1dc8364c9485d0`。逐文件 SHA-256、导入清单和插图出处保存在 `research/seedance-import/`。

- 公开能力：沿用原材料截至2026-11-22的网页阅读记录。本次未刷新产品、API可用性或价格。
- 基础方法：通用原理及已公开论文的局部机制，保留公式和说明。
- 替代基座：原系列 LTX-2 的19B、48层、4096/2048宽度等用于公开替代实现，不是 Seedance 的参数披露。
- 候选设计：12处512维条件支路、数据配比、学习率、因果派生等为可审阅方案，尚未训练验证。
- 算术示例：注意力、缓存、训练与交付费用是给定输入下的估算，不是 GPU 实测或实际报价。

28个模块导入了278个数学节点（含行内公式）、36张表、24段正文代码/流程块，以及16张论文裁图；另附16份教学机制代码。数学导读的公式和代码附录不计入上述导入统计。论文裁图保留原意、来源ID、原PDF哈希、页序和坐标，未复制整篇论文、模型权重或第三方完整源码。

原书中的 ElegantBook 示例、Lebesgue 积分练习及计量经济学模板附录移至 `examples/legacy-template.tex`，未丢弃。参考文献输出只保留原登记中的两条扩散相关文献，避免把模板中的经济学示例印入成品。原书的未完成小节保留；本次不是对旧稿的全面科学审校。

编译和阅读时顺便修正了原书的缺参数链接、越界 LCD 插图、数学模式内中文逗号，以及数处明显的 Fokker–Planck/OU 排版与公式问题：前向方程左侧为时间导数、一般瞬态方程不应再令其等于零、OU 漂移符号、转移均值和方差、稳定方差的二分之一因子、逆伽马分布名称与存在条件。其它旧稿结论未因此被视为已审核。

## 编译

需要 TeX Live（已在2025版验证）、XeLaTeX、latexmk；完整书稿参考文献使用 Biber。Consolas 不存在时自动退回 Latin Modern Mono，代码中的中文使用 TeX Live 的 Fandol 字体。

```powershell
# 两册全部编译，更新根目录PDF
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_books.ps1

# 仅专题 / 仅完整书稿
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_books.ps1 -Target seedance
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_books.ps1 -Target full
```

直接编译已有 TeX 不依赖来源仓库、Pandoc或Python。中间产物位于 `build/Seedance25/` 与 `build/StabelDiffusion/`，根目录PDF为交付文件。

需要重新导入原材料时才运行：

```powershell
python -m pip install pypandoc_binary
python scripts/import_seedance.py --source ../multimodal-journey
```

此操作会覆盖生成的 `m00.tex`–`m27.tex`、机制代码及来源章节；`primer.tex` 是手工新增的数学导读，不会被覆盖。已经直接修改TeX时，应先检查差异，避免重新导入覆盖本地改稿。

## 本地检查

```powershell
python -m pip install pymupdf pillow
python scripts/review_books.py
```

渲染结果、全书缩略图和结构报告输出至 `build/review/`。检查覆盖模块完整性、缺字、越界文字、Overfull、引用及内部链接；缩略图用于全页巡视，再放大检查公式、密集表格、插图与中文代码。编译/格式检查与模型效果验证是两回事。本次未运行教学机制代码、下载权重、训练模型或测量业务效果。
