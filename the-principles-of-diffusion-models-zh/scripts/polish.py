"""Explicit editorial corrections made after the cached initial translation.

Run after finalize.py. These edits affect prose and typography only.
"""
import re
from pathlib import Path
from translate import ROOT, DEST, TEX_FILES, protect, PAT

CORRECTIONS = {
 '能源型型号': '能量模型', '规范化流程': '归一化流',
 '变化视角': '变分视角', '分配级蒸馏': '分布层面的蒸馏',
 '流映射水平蒸馏': '流映射层面的蒸馏', '流量': '流',
 '发行到发行翻译的序言': '分布到分布传输导论',
 '密度传输骨干网': '密度传输的核心框架',
 '最佳传输': '最优传输', '最佳运输': '最优传输',
 '最佳传输配方': '最优传输表述', '变体最佳运输配方的关系': '不同最优传输表述之间的关系',
 '薛定谔电桥': '薛定谔桥', '最佳映射': '最优映射', 'OT 图': 'OT映射',
 'SB 和 EOT（双）等效': 'SB与EOT的双向等价',
 '全局推进和局部动态': '全局推前与局部动力学',
 '规范线性流程和回流': '标准线性流与reflow',
 '纠正流程': '整流（Rectified Flow）',
 '条件直线度并不意味着边际流量直线度': '条件轨迹平直并不意味着边缘流平直',
 '边际流': '边缘流', '边际': '边缘',
 '条件概率路径优先构造': '先构造条件概率路径',
 '条件流优先构造': '先构造条件流',
 '概率路径优先与流程优先构造': '先构造概率路径与先构造流的比较',
 '扩散模型、一般流匹配和节点的比较': '扩散模型、一般流匹配与NODE的比较',
 '免训练指导': '免训练引导', '辅助损耗': '辅助损失',
 '从强化学习到模型对齐的直接偏好优化': '用于模型对齐的强化学习与直接偏好优化',
 '布拉德利特里视图': 'Bradley--Terry视角',
 '（可选）基本规则': '（选读）基本结论',
 '（可选）为什么 DSM 要去噪：当然': '（选读）为什么DSM能够去噪：SURE',
 '具有评分功能的 Langevin 采样': '利用分数函数进行Langevin采样',
 '使用 Langevin Dynamics 进行采样': '利用Langevin动力学进行采样',
 '评分匹配': '分数匹配', '评分SDE': '分数SDE',
 '动机：分数是多少？': '动机：什么是分数函数？',
 '反向降噪过程': '反向去噪过程', '连续时间限制': '连续时间极限',
 '离散时间过渡': '离散时间转移', '从时间步进到时间并行求解器': '从逐步推进到时间并行求解器',
 '流映射的见解': '流映射的观点', '一般分歧': '一般散度',
 '普通电火花加工特例': '标准EDM的一个特例',
 '基于快速扩散的生成器': '快速扩散生成器',
 'Intuition for It\\^o Integration.': 'It\\^o积分的直观理解。',
 '导出一维\\^泰勒展开式的 o 公式': '由泰勒展开推导一维It\\^o公式',
 '测度的变量变化': '测度变换', '匹配边缘': '匹配边缘分布',
 'A 阐明扩散模型中训练损失的路线图': '理解扩散模型训练损失的路线图',
 'DPM-求解器': 'DPM-Solver', 'DPM 求解器': 'DPM-Solver',
 'E利用泰勒展开式估计积分': '利用泰勒展开估计积分',
 '斯坦因的恒等式': 'Stein恒等式', '黑森州': 'Hessian矩阵',
 '（可选）': '（选读）', '(可选)': '（选读）',
}


def main():
    for name in TEX_FILES:
        p = DEST/name
        s = p.read_text(encoding='utf-8')
        masked, pieces = protect(s)
        for a,b in sorted(CORRECTIONS.items(), key=lambda x: -len(x[0])):
            masked = masked.replace(a,b)
        masked = masked.replace('\u200b','')
        s = PAT.sub(lambda m:pieces[int(m[1])], masked)
        # Itô's TeX accent is deliberately kept in the original source. Fix the
        # surrounding heading spelling without touching mathematical content.
        s = s.replace('导出一维\\^泰勒展开式的 o 公式', '由泰勒展开推导一维It\\^o公式')
        s = s.replace('Intuition for It\\^o Integration.', 'It\\^o积分的直观理解。')
        s = s.replace('\\textbfs{法律}', '\\textbfs{概率律}')
        # The padding in the source's p-column tables exceeds the text block.
        s = s.replace('p{0.41\\linewidth}', 'p{0.40\\linewidth}')
        s = s.replace('p{0.40\\linewidth} p{0.40\\linewidth}',
                      'p{0.39\\linewidth} p{0.39\\linewidth}')
        # Let TeX break the long inline time-average identity at its equality.
        s = s.replace('$\\beta\\sum_{s=1}^T\\Delta_s\n=\n',
                      '$\\beta\\sum_{s=1}^T\\Delta_s\n\\allowbreak=\\allowbreak\n')
        s = s.replace('正向/逆时 SDE 和 PF-ODE 中的匹配边缘',
                      '前向与反向SDE及PF-ODE的边缘分布一致性')
        s = s.replace('观察到\n$\\beta', '可得：\\par\\noindent\n$\\beta')
        s = s.replace('\\subparagraph{1. 不受标准化常数的影响。}',
                      '\\subparagraph{1. 不受归一化常数的影响。}\\mbox{}\\\\')
        s = s.replace('\\paragraph{NCSN 的训练目标。}',
                      '\\paragraph{NCSN的训练目标。}\\mbox{}\\\\')
        s = s.replace('它们满足插值条件\n$P_1', '它们满足插值条件：\\newline\n$P_1')
        s = s.replace('t_{i-1})$ 和\n$P_1', 't_{i-1})$，\\newline\n$P_1')
        s = s.replace('这种单步构建使用两种模型评估：一种是在', '这种单步构造需要两次模型求值，分别位于')
        s = s.replace(' 和预测阶段的一个\n', '以及预测中间阶段\n')
        # This original reference is inside mathrm, which cannot paint Chinese.
        s = re.sub(r'\\mathrm\{(\\Cref\{[^}]+\})\}', r'\\text{\1}', s)
        s = re.sub(r'(\\[A-Za-z@]+\*?)(?=[\u4e00-\u9fff])',r'\1 ',s)
        p.write_text(s,encoding='utf-8')
    n = DEST/'notations.tex'
    n.write_text((ROOT/'editorial/notations.tex').read_text(encoding='utf-8'),encoding='utf-8')


if __name__ == '__main__': main()
