# 深度学习符号速查表
### 车同轨，书同文

*一份统一各教材与论文中常见深度学习符号的速查表。*

---

## 核心思想

在深度学习中，符号并不统一。不同作者可能用不同符号表示同一概念。例如，模型参数可以写作：

$$\theta,\quad w,\quad W,\quad \phi,\quad \psi,\quad \omega.$$

因此，最重要的习惯不是死记符号，而是随时问自己：

$$\text{这个符号代表的是什么？}$$

---

## 常见深度学习符号

| 符号 | English | 中文说明 | 常见替代符号 |
|:---:|---|---|---|
| $\theta$ | Model parameters | 模型所有可学习参数 | $w, W, \phi, \psi, \omega$ |
| $w$ | Weight | 单个可学习标量参数 | $\theta, \beta$ |
| $W$ | Weight matrix | 可学习权重矩阵 | $\Theta, A$ |
| $b$ | Bias | 可学习偏置项 | $c, \beta_0$ |
| $\phi$ | Encoder or feature extractor parameters | 模型某一部分（如编码器）的参数 | $\theta_e, \psi$ |
| $\psi$ | Auxiliary model parameters | 常用于评论网络、价值网络等辅助模块 | $\theta_v, \omega$ |
| $\omega$ | Generic parameters | 可学习参数的另一种写法 | $\theta, w$ |
| $x$ | Input | 输入数据 | $X, \mathbf{x}$ |
| $y$ | Target label | 正确答案／真实标签 | $t, r$ |
| $\hat{y}$ | Prediction | 模型的预测输出 | $y_{\text{pred}}, \tilde{y}$ |
| $f_\theta$ | Model / neural network | 以 $\theta$ 为参数的函数（即模型） | $h_\theta, g_\theta, M_\theta$ |
| $p_\theta(y \mid x)$ | Probabilistic model | 模型预测的条件概率分布 | $P(y \mid x;\theta)$ |
| $\mathcal{L}$ | Loss function | 损失函数，衡量预测有多差 | $L, J, \text{cost}$ |
| $J(\theta)$ | Objective function | 待最小化或最大化的目标函数 | $\mathcal{L}(\theta)$ |
| $\nabla_\theta \mathcal{L}$ | Gradient | 梯度，表示损失关于参数的变化方向 | $\text{grad}$ |
| $\eta$ | Learning rate | 训练时的步长（学习率） | $\alpha, \text{lr}$ |
| $\alpha$ | Learning rate or coefficient | 步长或加权系数 | $\eta, \lambda$ |
| $\lambda$ | Regularization strength | 正则化强度（惩罚项系数） | weight decay coefficient |
| $\beta$ | Momentum coefficient or parameter | 优化器中的动量系数，或统计参数 | decay rate |
| $z$ | Logit or latent variable | 激活前的原始分数，或隐变量 | latent code |
| $h$ | Hidden representation | 内部特征向量（隐层表示） | embedding, hidden state |
| $a$ | Activation or action | 神经网络中指激活值；强化学习中指动作 | $h$ |
| $\sigma$ | Activation function | 非线性激活函数 | sigmoid, activation |
| $D$ 或 $\mathcal{D}$ | Dataset | 训练样本的集合 | training set |
| $n$ 或 $N$ | Number of samples | 数据集大小或样本数量 | $m$ |
| $B$ | Batch size | 每次参数更新使用的样本数 | mini-batch size |
| $i$ | Index | 第 $i$ 个样本的索引 | sample index |
| $t$ | Time step or target | 序列模型中指时间步；监督学习中指目标值 | step |
| $T$ | Sequence length | 序列总时间步数 | horizon |
| $d$ | Dimension | 输入或特征向量的维度 | feature dimension |
| $K$ | Number of classes | 类别总数 | $C$ |
| $C$ | Classes or channels | 分类任务中指类别数；CNN中指通道数 | $K$ |
| $\mathbb{R}^d$ | Real vector space | $d$ 维实数向量空间 | shape notation |
| $\mathbb{E}$ | Expectation | 某分布下的期望值 | mean |
| $\sim$ | Sampled from | 表示随机变量服从某分布 | drawn from |
| $\propto$ | Proportional to | 相差一个常数倍（成正比） | proportional |
| $\arg\max$ | Argument of maximum | 取使函数值最大的输入 | best choice |
| $\arg\min$ | Argument of minimum | 取使函数值最小的输入 | optimizer target |
| $\theta^*$ | Optimal parameters | 训练后的最优参数 | $\hat{\theta}$ |
| $\hat{\theta}$ | Estimated parameters | 从数据中估计得到的参数 | $\theta^*$ |
| $q_\phi$ | Approximate distribution / encoder | 变分推断／VAE 中的近似分布（编码器） | inference model |
| $p_\theta$ | Generative model / decoder | 以 $\theta$ 为参数的生成模型（解码器） | likelihood model |
| $\pi$ | Policy or probability distribution | 强化学习中的策略；也可指概率分布 | $p, q, \mu$ |
| $s$ | State | 强化学习中的状态 | observation |
| $a$ | Action | 智能体采取的动作 | control |
| $r$ | Reward | 强化学习中的奖励信号 | reward value |
| $Q(s,a)$ | Action-value function | 在状态 $s$ 下执行动作 $a$ 的期望回报 | critic |
| $V(s)$ | Value function | 从状态 $s$ 出发的期望回报 | state value |
| $\gamma$ | Discount factor | 折扣因子，控制未来奖励的权重 | discount |
| $\tau$ | Trajectory or temperature | 强化学习中指轨迹；softmax 中指温度系数 | path, temp |
| $\mathcal{N}(\mu,\Sigma)$ | Normal distribution | 均值为 $\mu$、协方差为 $\Sigma$ 的高斯分布 | Gaussian |
| $\mu$ | Mean | 分布的均值或中心 | expectation |
| $\Sigma$ | Covariance matrix | 变量的方差与相关性 | covariance |
| $\epsilon$ | Small constant or noise | 用于数值稳定性的小常数，或随机噪声 | noise |
| $D_{\mathrm{KL}}$ | KL divergence | 两个分布之间的差异（KL 散度） | relative entropy |
| CE | Cross entropy | 常用分类损失（交叉熵） | log loss |
| MSE | Mean squared error | 常用回归损失（均方误差） | $\ell_2$ loss |
| NLL | Negative log-likelihood | 基于概率模型的损失（负对数似然） | log loss |
| SGD | Stochastic gradient descent | 基础梯度优化器（随机梯度下降） | optimizer |
| Adam | Adaptive optimizer | 常用自适应优化器 | optimizer |

---

## 常见公式模式

### 模型预测

$$\hat{y} = f_\theta(x)$$

含义：

$$\text{输入 } x \quad \longrightarrow \quad \text{模型 } f_\theta \quad \longrightarrow \quad \text{预测 } \hat{y}.$$

其中 $\theta$ 是模型的可学习参数。

### 训练目标

$$\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)$$

含义：找到使损失函数 $\mathcal{L}(\theta)$ 最小的最优参数 $\theta^*$。

### 梯度下降更新

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

含义：

$$\text{新参数} = \text{旧参数} - \text{学习率} \times \text{梯度}.$$

### 编码器与分类器示例

$$z = f_\phi(x), \qquad \hat{y} = g_\theta(z)$$

含义：

$$f_\phi = \text{参数为 } \phi \text{ 的编码器},$$

$$g_\theta = \text{参数为 } \theta \text{ 的分类器或解码器}.$$

$\phi$ 和 $\theta$ 都是可学习参数，但分别属于模型的不同部分。

---

## 如何在论文中读懂符号

遇到陌生符号时，依次问自己：

| 问题 | 常见符号 |
|---|---|
| 这是数据吗？ | $x, y, \mathcal{D}, x_i, y_i$ |
| 这是可学习参数吗？ | $\theta, w, W, b, \phi, \psi, \omega$ |
| 这是模型或函数吗？ | $f, g, h, p_\theta, q_\phi$ |
| 这是损失或目标函数吗？ | $L, \mathcal{L}, J$ |
| 这是超参数吗？ | $\eta, \alpha, \lambda, \beta, \gamma$ |
| 这是概率分布吗？ | $p, q, \pi, \mathcal{N}$ |
| 这是强化学习符号吗？ | $s, a, r, \pi, Q, V, \gamma$ |

---

## 核心结论

最重要的一条规则：

$$\boxed{\theta,\ w,\ W,\ \phi,\ \psi,\ \omega \text{ 往往都表示可学习参数。}}$$

作者使用不同字母，主要是为了区分模型的不同部分。例如：

$$p_\theta(x \mid z), \qquad q_\phi(z \mid x)$$

通常含义为：

$$p_\theta = \text{解码器或生成模型},$$

$$q_\phi = \text{编码器或近似推断模型}.$$

$\theta$ 和 $\phi$ 都是可学习参数，但属于不同的网络。

