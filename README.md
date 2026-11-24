# AIGC
扩散模型算法基础文档、训练、实验、部署等仓库。待更。

## 本地书稿（2026-11-24）

- [完整 AIGC 书稿 PDF](StabelDiffusion.pdf) / [TeX 主文件](StabelDiffusion.tex)：保留扩散基础，接入 Seedance 专题。
- [Seedance 2.5 专题 PDF](Seedance25.pdf) / [TeX 主文件](Seedance25.tex)：适合先检查本次新增内容。
- [阅读与编译说明](docs/seedance-book.md)：章节导航、来源边界、编译命令和本次修改说明。

专题来源于 [multimodal-journey](https://github.com/foocker/multimodal-journey) 的 M00–M27，包含数学导读、VAE、流匹配、DiT、位置编码、联合音视频、多参考、编辑、长上下文、训练、推理和完整推测方案。公开产品信息与候选工程设计分别标记，尚未进行模型训练或业务验证。

使用 TeX Live 2025 / XeLaTeX，在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_books.ps1
```

成品写到仓库根目录，编译中间文件保存在忽略的 `build/`。只编译专题可加 `-Target seedance`。

- 组合代码，都会附上原始出处，代码块里面，会附上论文出处。
- 最近时间有点尴尬，很难更新。。
