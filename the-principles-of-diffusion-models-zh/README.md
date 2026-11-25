# 扩散模型原理：中文初译

原著 **The Principles of Diffusion Models**，Chieh-Hsin Lai、Yang Song、Dongjun Kim、Yuki Mitsufuji、Stefano Ermon 著。

- 原著网站：<https://the-principles-of-diffusion-models.github.io/>
- 固定版本：arXiv:2510.21890v3，2026年8月27日。
- 官方摘要页：<https://arxiv.org/abs/2510.21890v3>
- 官方 TeX：<https://arxiv.org/src/2510.21890v3>
- 本地整理日期：2026年11月25日。

## 阅读与编辑

- [中文 PDF](principles-diffusion-models-zh.pdf)
- [中文 TeX 主文件](main_zh.tex)
- [分章中文源码](zh/)
- [官方英文 TeX 源码](original/tex/)
- [官方英文 PDF](original/principles-diffusion-models-v3-en.pdf)
- [术语表](glossary.md)
- [来源与文件清单](manifest.json)
- [编译与完整性检查](verification.json)

这是覆盖全书的**非官方中文初译**，包括前言、符号说明、12章正文、附录A–D、图注、表格和算法文字。原始配图内部文字、数学表达式和参考文献条目保留原语；TeX绘制的部分结构图使用了中文标题。

初译由 Google Translate 生成。本轮对照英文重校了前言和第一章，检查全书术语、标题及重点段落，并修订了符号说明和若干图注。部分正文仍保留初译语序，逐段学术审校尚未完成。

## 章节

1. 深度生成模型
2. 变分视角：从VAE到DDPM
3. 分数视角：从能量模型到NCSN
4. 当代扩散模型：分数SDE框架
5. 流视角：从归一化流到流匹配
6. 扩散模型的统一与系统视角
7. 扩散模型与最优传输（选读）
8. 引导与可控生成
9. 扩散快速采样的数值求解器
10. 基于蒸馏的快速采样方法
11. 从头学习快速生成器
12. 结语与展望

附录 A 介绍微分方程，附录 B 介绍密度演化，附录 C 介绍伊藤微积分与吉尔萨诺夫定理，附录 D 汇集补充材料与证明。文件保留官方命名，正文文件名的编号比印刷章节号小一。

## 编译

依赖 TeX Live 2025、XeLaTeX、latexmk、Biber 与 Windows 中文字体。在本目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1
```

PDF写到本目录，编译中间文件在 `build/latex/`。日常编辑 `main_zh.tex` 或 `zh/` 下的分章文件后直接编译即可；编译不会再次联网翻译。

## 初译生成记录

以下命令用于复现本次初译，**会覆盖中文源码**。人工修订后不要直接重放，应先保存修订副本。

```powershell
python scripts/translate.py
python scripts/prepare_edition.py
python scripts/finalize.py
python scripts/polish.py
powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1
python scripts/verify.py
```

翻译过程将公式、标签、引用标识、图片路径和TeX控制结构替换为占位符后处理正文，再校验并恢复；遇到占位符缺失或重排时，改为逐个文本片段翻译。缓存位于忽略的 `build/translation-cache/`，可断点续跑。这里的并发仅为翻译HTTP请求。

`source-integrity.json`记录初译恢复阶段的TeX结构核验，`translation-report.json`记录章节与字数。最终稿另外做了标题、术语、符号表和排版编辑；最终检查见 `verification.json`，这些结构检查不等同于人工语义审校。

## 来源与署名

原作者、原始书名、官方网站与版本信息均保留。`original/`中的TeX压缩包和解包源码不作修改。原著在arXiv标注的许可为 [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)。本目录是本地阅读用翻译整理，并非作者或出版社发布的中文版。
