"""Apply audited local edits to the original theory; never synthesize a replacement."""
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '73b7b7adfe36e6d2a828d2274ad0b4b52a20db9e'
full = subprocess.check_output(['git', 'show', BASE_COMMIT + ':StabelDiffusion.tex'], cwd=ROOT).decode('utf-8')
start = full.index('\\part{')
chapters = list(re.finditer(r'\\chapter\{', full[start:]))
baseline = full[start:start + chapters[4].start()]
s = baseline
changes = []

def replace(old, new, reason, count=1):
    global s
    assert s.count(old) == count, (reason, s.count(old), old)
    s = s.replace(old, new)
    changes.append({'reason': reason, 'old': old, 'new': new, 'occurrences': count})

def after(anchor, fragment):
    replace(anchor, anchor + '\n\n\\input{chapters/theory/additions/' + fragment + '}\n',
            'Insert local supplement: ' + fragment)

replace(r'\item 离散扩散过程变量：', r'\item 离散扩散过程变量：$x_0,x_1,\ldots,x_T$。', 'Complete empty symbol')
replace(r'\item probability flow ODE', r'\item probability flow ODE：与指定前向 SDE 共享逐时边缘密度的确定性流，见连续版 Score SDE。', 'Complete empty symbol')
replace(r'\item ELBO ', r'\item ELBO：证据下界，最大化它；若作为损失最小化，则使用它的负值。', 'Complete empty symbol')
after(r'\section{基本介绍}', 'intro')
replace('KL散度恒大于0', 'KL散度非负', 'KL can equal zero')
replace('从中了解到，VLB和ELBO在定义上存在一个正负符号只差。',
        r'这里要核对的是目标的符号：ELBO 是对数似然的下界；本文后面最小化的 $L_{\rm VLB}$ 是负 ELBO，即负对数似然的上界。VLB 的命名在不同资料中并不完全统一。', 'Distinguish ELBO and negative training objective')
after(r'重参数技巧：\href{https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick}{reparameterization-trick}', 'gaussian-reparam')
after('关于先验，后验，极大似然，似然函数等', 'bayes')
after("定义2.1给出了概率分布$p$的Stein score function。该定义从Stein 1972年写的一篇文章中抽取出来，相关内容可参考\\href{https://en.wikipedia.org/wiki/Stein%27s_method}{Stein's method}。", 'score-definitions')
after(r'\href{https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo}{Markov chain Monte Carlo}', 'mcmc')
replace('增加一个恒大于0的变量', '增加一个非负的 KL 项', 'Nonnegative variational gap')
replace(r'数据的负对数似然为$-log_{p_{\theta}}(x_0)$', r'数据的负对数似然为$-\log p_{\theta}(x_0)$', 'Correct likelihood notation')

# Two originally misplaced displays are retained and moved to their existing relevant headings.
att_start = s.index('\\begin{equation}\n  \\begin{aligned} & \n    \\text { Attention }')
att_end = s.index('\\end{equation}', att_start) + len('\\end{equation}')
attention = s[att_start:att_end]
s = s[:att_start] + s[att_end:]
bound_start = s.index('\\begin{equation}\n  \\begin{aligned} \\mathbf{KL}')
bound_end = s.index('\\end{equation}', bound_start) + len('\\end{equation}')
bound = s[bound_start:bound_end]
s = s[:bound_start] + s[bound_end:]
changes.append({'reason': 'Move the unchanged cross-attention display to the original modules heading and the unchanged score-KL bound to the original theoretical-results heading.', 'moved_displays': 2})
after('整个过程是比较简单的，核心在等式第五行的一个贝叶斯转换，转换的原因在于给定扩散过程，逆向转移在给定$x_0$的基础上，能被具体的计算出来。',
      'elbo-bridge')
replace('和随机变量$ X_n$的二阶矩有限', '和随机变量$ X$的二阶矩有限', 'Correct limit variable')
replace('由切比雪夫不等式可知：', '对非负随机变量 $|X_n-X|^2$ 使用 Markov 不等式可知：', 'Use the valid uncentered second-moment bound')
replace(r'\frac{E\left|X_n-X\right|^2-E^2\left|X_n-X\right|}{\varepsilon^2}', r'\frac{E\left|X_n-X\right|^2}{\varepsilon^2}', 'Correct probability bound')
replace(r'\text { l.i.m } \mathrm{m}_{\Delta t 0}', r'\operatorname{l.i.m}_{\Delta t\to0}', 'Repair mean-square continuity notation')
replace('在$x_0$处均方连续', '在$t_0$处均方连续', 'Time coordinate in continuity')
replace(r'    \Delta t \rightarrow 0, \forall \varepsilon, \forall \eta, P\left(\left|X\left(t_0+\Delta t\right)-X\left(t_0\right)\right|>\varepsilon\right)<\eta',
        r'    \operatorname{l.i.m}_{\Delta t\to0}\frac{X(t_0+\Delta t)-X(t_0)}{\Delta t}=Y', 'Restore missing difference quotient in derivative definition')
replace(r'a_n Y^{(n)}(t)+a_n Y^{(n)}(t)', r'a_n Y^{(n)}(t)+a_{n-1}Y^{(n-1)}(t)', 'Correct repeated highest derivative')
replace(r'若$X(t)$在$[a, b]$ 上均方可积，则：' + '\n' + r'  \begin{equation}' + '\n' + r'    \left[\int_a^t X(s) d s\right]^{\prime}=X(t)',
        r'若$X(t)$在$[a, b]$ 上均方连续，则：' + '\n' + r'  \begin{equation}' + '\n' + r'    \left[\int_a^t X(s) d s\right]^{\prime}=X(t)', 'Add sufficient continuity assumption to fundamental theorem')
replace(r'$X(t)$表示时间$t$时刻质点的位置，则$X(t)=\Delta x(X_1 + X_2 + \dots + X_{[\frac{t}{\Delta t}]})$，其中$[x]$表示的超过$x$',
        r'$X(t)$表示时间$t$时刻质点的位置，则$X(t)=\Delta x(X_1 + X_2 + \dots + X_{[\frac{t}{\Delta t}]})$，其中$[x]$表示不超过$x$', 'Repair floor definition')
after(r'  若$c=1$，则称为标准布朗运动$B(t)\sim N(0, t)$。', 'brownian-conditions')
replace(r'e^{-\frac{1}{2}(x-\mu)^2}', r'e^{-\frac{(x-\mu)^2}{2\sigma^2}}', 'Restore variance in Gaussian density')
replace(r'3 \sigma^4+\mu \sigma^2+5 \mu^2+\mu^4', r'3\sigma^4+6\mu^2\sigma^2+\mu^4', 'Correct Gaussian fourth moment')
replace(r'2 \sigma^4+\left(\mu-2 \mu^2\right) \sigma^2+5 \mu^2', r'2\sigma^4+4\mu^2\sigma^2', 'Correct variance of squared Gaussian')
replace(r'& =\lim _{\delta \rightarrow 0} 2 \delta T', r'& \leq\lim _{\delta \rightarrow 0} 2 \delta T=0', 'Variance bound is an inequality')
replace(r'\right|^2-\left.T\right|^2=\lim _{n \rightarrow \infty} 2 \delta T=0', r'\right|^2-\left.T\right|^2\leq\lim _{\delta \rightarrow 0} 2 \delta T=0', 'Use variance upper bound in quadratic variation conclusion')
replace(r'式3.26中，将等式右边$T$看作$dt \text{在}, [0, T]$上的积分，等式左边$W(t_{i+1} - W(t_i))$记为$dW \text{或}dW_t$，则',
        r'上面的二次变差等式右边可写成 $\int_0^T\mathrm dt$。将增量 $W(t_{i+1})-W(t_i)$ 记为 $\mathrm dW$，便得到积分意义下的运算规则：', 'Repair differential notation and stale equation reference')
replace('直观的理解，（3.33）实际就是 $f(t, X_t)$的二元泰勒展开，只展开到这几项是因为根据（3.27,28）等几个结论，',
        '直观地看，上面的 Itô 引理就是 $f(t,X_t)$ 的二元 Taylor 展开；保留二次项，是因为布朗增量的平方在积分极限下仍有贡献，', 'Replace stale Ito references with actual reasoning')
after('得到含有布朗运动微分的Itô随机微分方程和扩散方程。', 'ito-bridge')
replace(r'  设$\{X(t), t\in [0, T] \}$为随机过程，若积分:',
        r'  设被积过程 $g(X(t),t)$ 对布朗运动的滤过适应，并满足 $\mathbb E\int_0^T|g(X(t),t)|^2\,\mathrm dt<\infty$，若积分:', 'State adaptedness and square integrability for Ito integral')
replace(r'p(t, x)+\frac{\partial^2}{\partial x^2}\left(\frac{1}{2} \sigma^2 p(t, x)\right)\right] d x d t',
        r'(\mu p(t, x))+\frac{\partial^2}{\partial x^2}\left(\frac{1}{2} \sigma^2 p(t, x)\right)\right] d x d t', 'Restore drift in FPE intermediate sum')
replace(r'$u(t, x)=\mathbb{E}_x\left[f\left(X_T\right)\right]=\mathbb{E}\left[f\left(X_t\right) \mid X_t=x\right]$',
        r'$u(t,x)=\mathbb E[f(X_T)\mid X_t=x]$，且 $u(T,x)=f(x)$', 'Correct terminal conditional expectation')
for old, new in [
    (r'0=\frac{d}{d t}\langle p, f\rangle=\left\langle\partial_t p, f\right\rangle+\left\langle p, \partial_t f\right\rangle', r'0=\frac{d}{d t}\langle p, u\rangle=\left\langle\partial_t p, u\right\rangle+\left\langle p, \partial_t u\right\rangle'),
    (r'=\left\langle\partial_t p, f\right\rangle-\langle p, L f\rangle=\left\langle\partial_t p, f\right\rangle-\left\langle L^* p, f\right\rangle', r'=\left\langle\partial_t p, u\right\rangle-\langle p, L u\rangle=\left\langle\partial_t p, u\right\rangle-\left\langle L^* p, u\right\rangle'),
    (r'=\left\langle\partial_t p-L^* p, f\right\rangle', r'=\left\langle\partial_t p-L^* p, u\right\rangle')]:
    replace(old, new, 'Use terminal-value u consistently in duality argument')
replace(r'我们得到Kolmogorov前向方程 $\partial_t p-L^* p$', r'由任意测试函数的弱形式可得 Kolmogorov 前向方程 $\partial_t p-L^*p=0$', 'Complete forward equation and weak-form conclusion')
after(r'p(x)=\frac{1}{\sigma(x)^2} \exp \left(\int \frac{2 \mu(x)}{\sigma^2(x)} d x\right)'+'\n$$', 'stationary-normalizer')
after('其中$p_d$是$p_{data}$的简写。', 'energy-note') if '其中$p_d$是$p_{data}$的简写。' in s else None

# Preserve the historical list, clarifying its limitations locally rather than rewriting it.
replace(r'\section{整体认识}', r'\section{整体认识}'+'\n这些条目保留了阅读早期论文时的摘录和疑问。关于训练与似然、连续与离散、采样步数的表述，后文会逐项补充适用条件；其中的实验步数和硬件记录应按当时设置理解。', 'Context for historical reading notes')
replace('基于分数的生成模型，只能用在连续数据上，若离散数据，可先利用VAE将其映射到连续的隐空间中，',
        '上述使用空间梯度的分数模型直接作用于连续变量；若处理离散数据，一种做法是先利用VAE将其映射到连续的隐空间中，', 'Do not exclude native discrete diffusion')
replace('需要采用上面提出的方法。”', '可采用上面提出的方法来改善理论控制。”', 'Avoid exclusive claim about ODE likelihood training')
replace('它只适合训\n  练diffusion SDE的最大似然估计，但并不适合训练diffusion ODE的最大似然估计。',
        '相应似然加权目标可控制 diffusion SDE 的分布误差，但不能直接将同一个误差保证套到 diffusion ODE 上。', 'Correct historical quotation with local qualified paraphrase')
replace('SDE的模型是不能计算似然值的，但是ODE是可以精确就算的。优化一阶分数损失并不能得到ODE的似然，负对数似然被',
        '通常的扩散 SDE 不直接提供闭式数据似然，概率流 ODE 则可通过散度积分计算模型似然。仅优化一阶分数损失不自动保证 ODE 似然准确；相关界涉及', 'Distinguish likelihood computation from error control in historical note')
replace('通过sliced score matching 训练NCSN，即去噪分数匹配。', 'NCSN 主要使用去噪分数匹配；sliced score matching 是另一种估计方法。', 'Distinguish DSM and SSM')
replace('在连续化DDPM和SGM的时候，就去掉了DDPM的马尔科夫性的假设，', '连续化 DDPM 与 SGM 后得到的 Itô 扩散仍具有马尔可夫性；', 'Continuous diffusion is still Markov')
replace('添加也是从大噪声到小噪声，直到数据变为高斯分布。', '前向加噪从小尺度走向大尺度，生成采样才从大噪声走向小噪声。', 'Correct training-path versus sampling direction')
replace('容易看到，当$T \\rightarrow \\infty$时，$q(\\mathbf{x}_t|\\mathbf{x}_{t-1})$趋近于标准状态分布。',
        '当累积信号系数 $\\bar\\alpha_T\\to0$ 时，终端边缘 $q(\\mathbf x_T)$ 才趋近标准高斯；单步转移核并不因此变成标准高斯。', 'Correct terminal convergence statement')
replace(r'$\bar{\epsilon}_{t-2} = \sqrt{\alpha_{t}(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{1-\alpha_t}\epsilon_{t-1}$',
        r'$\bar{\epsilon}_{t-2} = [\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-2}+\sqrt{1-\alpha_t}\epsilon_{t-1}]/\sqrt{1-\alpha_t\alpha_{t-1}}$，归一化后仍为标准高斯', 'Normalize combined Gaussian noise')
replace(r'只剩下反向$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$', r'只剩下反向$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$', 'Correct reverse conditional direction')
replace('这两者相等，因为它们都是表示初始数据和中间数据的联合分布。', '这两者都定义了整条轨迹的联合分布，但并不会自动相等；学习的目标是让反向模型尽量逼近前向过程所定义的联合分布。', 'Do not assume learned and data joint distributions equal')
replace(r':=-L_{\mathbf{VLB}}\left(\mathbf{x}_0\right)', r':=L_{\mathbf{VLB}}', 'Consistent negative-ELBO loss sign')
replace('这里需要说明一下，很多地方说变分下界VLB和证据下界是一样的，包括Wikipedia，但实际上Wikipedia关于ELBO的定义和一些扩散论文对VLB的定义\n能看出来，两者差一个正负符号(这里面应该存在一些误用)。',
        '这里仍沿用前文的符号约定：最大化 ELBO，等价于最小化负 ELBO；后者在下面记为 $L_{\\rm VLB}$。阅读其它资料时，应以具体公式的正负号为准。', 'Explain sign convention locally')
replace(r"用式(4.8)中的$q(\mathbf{x}_t \mid \mathbf{x}_{t-1},\mathbf{x}_0)$替换一下即得到最终'可操作的'优化目标。",
        r'通过贝叶斯公式改写为含 $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ 的项，并将边缘比值连乘相消，即得到最终可计算的优化目标。', 'Explain posterior substitution and telescoping')
replace(r'\mathbf{x}_{t-1}=\mathcal{N}\left(\mathbf{x}_{t-1} ; \frac{1}{\sqrt{\alpha_t}}',
        r'p_\theta(\mathbf x_{t-1}\mid\mathbf x_t)=\mathcal{N}\left(\mathbf{x}_{t-1} ; \frac{1}{\sqrt{\alpha_t}}', 'Use distribution rather than sample on Gaussian-density lhs')
replace('经过简化(将不影响优化的系数等删去等)，最终得到', '进一步选择重新加权的简化训练目标，得到', 'Simple loss is reweighting')
after(r'\label{fig:DDPM-algo}'+'\n'+r'\end{figure}', 'ddpm-loss')
replace('另外2010年的一篇文章A Connection Between Score Matching and Denoising Autoencoders提出的Denoising score matching能做到0梯度计算(计算量更少了)达到等价的优化效果，但是是对扰动后的数据的密度估计。',
        '另外，2011年的文章 A Connection Between Score Matching and Denoising Autoencoders 提出的去噪分数匹配，避免了损失中对分数网络输入求导的迹计算；网络参数仍需正常反向传播。它估计的是扰动后的数据分数。', 'Correct DSM date and gradient claim')
replace(r'\mathcal{N}(\tilde{\mathbf{x}} \mid \mathbb{t}, \sigma^2I)'+'\n', r'\mathcal{N}(\tilde{\mathbf{x}} \mid \mathbb{t}, \sigma^2I)\,\mathrm d\mathbb t'+'\n', 'Complete Gaussian convolution integration measure')
after(r'\subsection{score matching}', 'score-matching')
replace('使得优化SM模型，能得到SDE对应的数据边缘分布，也就是原始数据的似然函数值。', '说明在指定权重与正则条件下，分数匹配误差可控制反向 SDE 的数据分布误差。这个界本身不是逐样本似然的计算公式。', 'Separate error bound from density evaluation')
replace('论文的网络是使用概率流网络来训练的?。', '这里的训练对象仍是分数网络；概率流 ODE 是由它进一步构造的。', 'Resolve original training question')
replace('论文证明了一阶分数模型并不能得到扩散对应的ODE的似然，该似然被一阶，二阶，三阶分数匹配误差所控制，以此提出了一个高阶的分数去噪模型，能得到更好的原始数据的似然逼近，并同时能保证生成高质量数据。',
        '相关分析表明，一阶分数匹配误差本身不足以无条件控制 ODE 的似然误差；在相应假设下，高阶分数误差也进入控制关系，由此提出高阶去噪分数匹配训练。似然改善与样本质量仍应分别评估。', 'Correct ODE likelihood interpretation')
after('似然改善与样本质量仍应分别评估。', 'likelihood')
after(r'\section{连续版Score SDE}', 'sde-notation')
replace(r'$\mathbf{x}_{\sigma_i}=\mathbf{x}_0+\sigma_i \mathbf{I}$', r'$\mathbf{x}_{\sigma_i}=\mathbf{x}_0+\sigma_i z$，$z\sim\mathcal N(0,I)$', 'Noise draw is a vector rather than identity matrix')
replace('当转移核是优化目标(47)的解的时候，若', '对这里预先规定的前向过程，若', 'Forward transition kernel is not an optimizer')
replace('均值和方程能具体计算出来', '均值和方差能通过相应方程计算出来', 'Correct variance wording')
after('% \\subsection{stochastic gradient Langevin dynamics}', 'probability-flow')
after('\\input{chapters/theory/additions/probability-flow}', 'parameterizations')
replace(r'\section{一些理论结果}', r'\section{一些理论结果}'+'\n分数匹配的误差在什么条件下能控制生成分布的误差？先看前面提到的这个界：\n'+bound+'\n\\input{chapters/theory/additions/guarantees}\n', 'Move and explain original KL bound')
after(r'\subsection{DDIM}', 'ddim-notation')
after('ANALYTIC-DPM一文的前向是DDIM的模式，但是在推导过程中，作者把均值用DDIM模型表达的式子替换为理论上等价的分数模型，因分数模型有具体的表达式，因此整个计算能走下去。', 'ddim')
replace('并利用变量替换法，给出了ODE的具体解，具体解的形式在ODE的数值求解中有充分的研究，因此能避免成百次的迭代，达到加速的效果。', '并利用变量替换法精确积分已知的线性部分，剩余网络积分再作数值近似，从而用更少的网络调用达到加速效果。', 'Distinguish integral representation and closed-form solution')
replace('增加分类引导(calssify-free guide)', '增加无分类器引导（classifier-free guidance）', 'Correct CFG name')
after('因为算法本身和训练无关，只用在采样生成样本中，因此法只需要不多的代码，就能实现，而且在实际中成为比较好的加速版的采样求解器。', 'solvers')
after(r'\subsection{蒸馏}', 'distillation')
replace(r'\nabla \log p_t(\mathbf{x})=-\mathbb{E}', r'\nabla_{\mathbf x_t}\log p_t(\mathbf{x}_t)=-\mathbb{E}', 'Evaluate score at the noisy coordinate')
replace(r'\mathcal{N}\left(\mathbf{x} ; t^2 \boldsymbol{I}\right)', r'\mathcal{N}\left(\mathbf{x}, t^2 \boldsymbol{I}\right)', 'Correct Gaussian mean-covariance notation')
replace(r'$c_{skip}(0)=1, c_{skip}(0)=0$', r'$c_{skip}(0)=1, c_{out}(0)=0$', 'Correct consistency output boundary')
after(r'\label{fig:LCD}'+'\n'+r'\end{figure}', 'consistency-updates')
after(r'\subsection{角色一致性}', 'identity')
after('\\input{chapters/theory/additions/identity}', 'discrete')
after(r'\subsubsection{VAE、CLIP、U-Net、U-ViT、DiT等模块}', 'modules')
after('\\input{chapters/theory/additions/modules}', 'attention-placeholder')
s = s.replace('\\input{chapters/theory/additions/attention-placeholder}', attention)
after(r'\subsection{Rectified Flow}', 'flow')
after(r'\subsubsection{条件控制}', 'guidance')
after(r'\section{应用}', 'historical-applications')
after(r'\subsubsection{SDXL}', 'scaling')
after(r'\subsubsection{LORA}', 'lora')
after(r'\subsection{3D重建}', 'geometry')
after(r'\subsubsection{SinMDM}', 'motion')
after(r'\subsection{Diffusers、SCEPTER}', 'frameworks')
after(r'\subsection{ComfyUI、WebUI、Fooocus}', 'workflows')
replace(r'\mathbf{Q}=\mathbf{W}_Q^{(i)} \cdot \varphi_i\left(\mathbf{z}_i\right), \mathbf{K}=\mathbf{W}_K^{(i)} \cdot \tau_\theta(y), \mathbf{V}=\mathbf{W}_V^{(i)} \cdot \tau_\theta(y)',
        r'\mathbf Q=\varphi_i(\mathbf z_i)\mathbf W_Q^{(i)},\quad\mathbf K=\tau_\theta(y)\mathbf W_K^{(i)},\quad\mathbf V=\tau_\theta(y)\mathbf W_V^{(i)}', 'Consistent row-token matrix multiplication in original attention display')
replace(r'\mathbf{W}_Q^{(i)} \in \mathbb{R}^{d \times d_e^i}, \mathbf{W}_K^{(i)}, \mathbf{W}_V^{(i)} \in \mathbb{R}^{d \times d_r}',
        r'\mathbf W_Q^{(i)}\in\mathbb R^{d_c^i\times d},\quad\mathbf W_K^{(i)},\mathbf W_V^{(i)}\in\mathbb R^{d_r\times d}', 'Correct projection shapes for row-token features')

# Add stable labels on separate lines so original formulas remain intact.
for formula, label in [
    (r'\mathbf{KL}\left(q\left(\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}_T\right) \| p_\theta\left(\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}_T\right)\right)', 'orig:path-kl'),
    (r'J(\boldsymbol{\theta}) \triangleq \mathbb{E}_{p_d}', 'orig:score-loss'),
    (r'\tilde{\mathbf{x}}_t=\tilde{\mathbf{x}}_{t-1}+\frac{\epsilon}{2}', 'orig:langevin'),
    (r'\boldsymbol{\theta}^*=\underset{\boldsymbol{\theta}}{\arg \min } \sum_{i=1}^N\left(1-\alpha_i\right)', 'orig:discrete-score')]:
    pos = s.index(formula)
    begin = s.rfind('\\begin{equation}', 0, pos)
    assert begin >= 0
    s = s[:begin] + s[begin:].replace('\\begin{equation}', '\\begin{equation}\n\\label{'+label+'}', 1)
    changes.append({'reason': 'Label an existing equation without changing the formula', 'label': label})
replace('从式(4.6)的推导中看到', r'从式\eqref{orig:path-kl}的推导中看到', 'Stable equation reference')
replace('优化目标(4.6)$ii$式', r'优化目标\eqref{orig:path-kl}的 $ii$ 项', 'Stable equation reference')
replace('去逼近式$(33)$的迹', r'去逼近式\eqref{orig:score-loss}中的迹', 'Stable equation reference')
replace('采样算法(对比式(38))', r'采样算法（对比式\eqref{orig:langevin}）', 'Stable equation reference')
replace('问题是(43)式能通过ELBO推导出来？', r'问题是式\eqref{orig:discrete-score}能通过ELBO推导出来？', 'Stable equation reference')
replace('得到(43)所示的分数表达形式', r'得到式\eqref{orig:discrete-score}所示的分数表达形式', 'Stable equation reference')
assert not any(c < 32 and c not in (9,10) for c in s.encode()), 'Unexpected control character in TeX source'

target = ROOT / 'chapters/theory/original-foundations.tex'
target.write_text(s, encoding='utf-8', newline='\n')
(ROOT / 'build/original-theory-baseline.tex').write_text(baseline, encoding='utf-8')
(ROOT / 'research/theory-update/local-corrections.json').write_text(
    json.dumps(changes, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
original_lines = [line for line in baseline.splitlines() if line.strip()]
new_counts = Counter(line for line in s.splitlines() if line.strip())
retained = sum(min(n,new_counts[line]) for line,n in Counter(original_lines).items())
heads = re.compile(r'^\\(?:part|chapter|section|subsection|subsubsection)\{.*$', re.M)
original_heads = heads.findall(baseline)
new_heads = heads.findall(s)
cursor = iter(new_heads)
assert all(any(n == h for n in cursor) for h in original_heads)
metadata = {
    'date': '2026-11-24',
    'base_commit': BASE_COMMIT,
    'base_fragment_sha256': hashlib.sha256(baseline.encode()).hexdigest(),
    'policy': 'Preserve original chapters, prose and detailed derivations; local corrections and in-place supplements only.',
    'original_nonempty_lines': len(original_lines),
    'original_nonempty_lines_retained_verbatim_in_foundations': retained,
    'retained_line_fraction': round(retained / len(original_lines), 5),
    'original_headings': original_heads,
    'original_heading_order_preserved': True,
    'local_edit_records': len(changes),
    'moved_displays_without_content_removal': 2,
    'attention_projection_dimensions_corrected': True,
    'supplements': sorted(re.findall(r'\\input\{chapters/theory/additions/([^}]+)\}',s)),
    'concise_rewrite_included': False,
}
(ROOT/'research/theory-update/restoration.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in metadata.items() if k not in ('original_headings','supplements')}, indent=2))
