# 核心术语

通用数学术语以工作区中的 math-translator 词典（`tools/math-translator/dictionaries`）为参照。词典未收录的扩散模型术语按上下文选译。篇章标题中的 *Deep Generative Modeling* 采用“深度生成模型”，讲建模过程时具体说明是在学习或逼近数据分布。统计物理与能量模型中的 *partition function* 沿用“配分函数”，用来指归一化常数 \(Z\)。

| 英文 | 本译本用语 |
|---|---|
| diffusion model | 扩散模型 |
| deep generative model (DGM) | 深度生成模型 |
| autoregressive model (AR) | 自回归模型 |
| likelihood | 似然 |
| maximum likelihood estimation (MLE) | 最大似然估计 |
| total variation distance | 全变差距离 |
| change of variables | 变量变换 |
| Jacobian matrix / determinant | 雅可比矩阵／雅可比行列式 |
| Hessian matrix | 黑塞矩阵 |
| Fréchet derivative | 弗雷歇导数 |
| coupling | 耦合 |
| forward corruption / perturbation | 前向扰动／加噪 |
| reverse-time SDE | 反向时间随机微分方程 |
| score / score function | 分数／分数函数，即对数密度的梯度 |
| score matching | 分数匹配 |
| denoising score matching (DSM) | 去噪分数匹配 |
| noise conditional score network (NCSN) | 噪声条件分数网络 |
| energy-based model (EBM) | 能量模型 |
| variational autoencoder (VAE) | 变分自编码器 |
| evidence lower bound (ELBO) | 证据下界 |
| normalizing flow (NF) | 归一化流 |
| neural ODE (NODE) | 神经常微分方程 |
| probability flow ODE (PF-ODE) | 概率流常微分方程 |
| flow matching (FM) | 流匹配 |
| conditional flow matching (CFM) | 条件流匹配 |
| flow map | 流映射 |
| marginal distribution | 边缘分布 |
| marginalization | 边缘化 |
| pushforward | 推前 |
| optimal transport (OT) | 最优传输 |
| entropy-regularized OT (EOT) | 熵正则化最优传输 |
| Schrödinger bridge (SB) | 薛定谔桥 |
| classifier guidance (CG) | 分类器引导 |
| classifier-free guidance (CFG) | 无分类器引导 |
| distillation | 蒸馏 |
| consistency model (CM) | 一致性模型 |
| consistency trajectory model (CTM) | 一致性轨迹模型 |
| mean flow (MF) | 平均流 |
| transition kernel | 转移核 |
| continuity equation | 连续性方程 |
| Fokker–Planck equation | 福克尔-普朗克方程 |
| change of measure | 测度变换 |
| Itô calculus | 伊藤微积分 |
| Radon–Nikodym derivative | 拉东-尼科迪姆导数 |
| Girsanov theorem | 吉尔萨诺夫定理 |

正文为初译，个别用语仍可能存在变体；应结合原文和数学定义理解。
