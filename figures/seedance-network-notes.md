# Seedance 2.5 推测网络图的结构依据

本组五页图针对 Seedance 系列演进所作的候选方案，不是 LTX 替代实现的逐层图，也不声称是已披露的 2.5 网络。

## 本次固定的图示方案

- 视觉表示采用 3D Causal VAE；不把 LTX 的 128 通道、时间/8、空间/32写成 Seedance 参数。
- Dual-MMDiT 在本图中明确指两个生成分支：视频--文本 MMDiT、音频--文本 MMDiT。两分支的文本流独立投影、分别更新；音视频状态再交换。这个具体连接是推测展开，公开的“基于 MMDiT 的双分支”不足以唯一确定它。
- 文本 token 产生 Q/K/V，与媒体 token 沿 token 维组合进联合注意力；输出按 query 行拆回，各有 MLP 与残差。
- 参考与源条件沿后文 M07/M17/M27 的独立读取/残差候选路径，不把所有控制压成 AdaLN。
- 噪声时间单列输入。图中 Norm 不指定是否自适应；省略具体时间参数化不等于断言模型没有 adaptive norm。
- 长上下文整段联合生成是基线；另训的因果续写另标，不混为同一个推理模式。

## 原始材料核对

| 依据 | 定位 | 支持范围 |
|---|---|---|
| [Seedance 1.0 v2](https://arxiv.org/abs/2506.09113v2) | PDF 第4页 §2.1--2.2；第5页 Figure 2 与统一任务描述 | 因果 VAE、去掉 DiT patchification；视觉/文本 token 拼接；空间 MMDiT、视觉时空注意力；各模态独立 adaptive layer norm/QKV/MLP；多镜头 MM-RoPE。它不能确认 2.5 的参数或层号。 |
| [Seedance 1.5 Pro v3](https://arxiv.org/abs/2512.13507v3) | PDF 第1页摘要、第2页贡献列表与 Figure 2 | dual-branch DiT、cross-modal joint module、基于 MMDiT 的联合音视频方向。未给足以唯一确定本图的内部连线。 |
| [Seedance 2.0 v1](https://arxiv.org/abs/2604.14148v1) | PDF 第1--3页 | 四模态参考、音视频联合生成、编辑与续写能力；未披露完整逐层架构。 |
| `chapters/seedance/m07.tex`、`m17.tex`、`m19.tex`、`m27.tex` | 参考读取、独立源条件塔、系统候选方案 | 参考 K/V + 门控残差；源条件独立编码、保留网格并投影到选定层；时间/主体/镜头绑定。 |
| `chapters/seedance/m19.tex`、`m21.tex` | LTX 替代主干、精确接口规格 | 用于另一个可实施替代路线；其固定文本 cross-attention、AdaLN 和精确层数不直接移到本组 Seedance 推测图。 |

“条件不走 AdaLN”在这里指文本与媒体的主要内容入口，不能扩写成“Seedance 从未使用 adaptive layer norm”。1.0 原文明确提到了它；2.5 的具体归一化与时间调制仍未确定。

只修改图源与独立图册，不改书稿段落与既有章节顺序。

编译：`xelatex -interaction=nonstopmode -halt-on-error -output-directory=build/seedance-architecture figures/seedance-network-standalone.tex`。
