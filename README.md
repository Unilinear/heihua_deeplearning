# Deep Learning Notation Cheat Sheet
### 车同轨，书同文

*A concise cheat sheet unifying common deep learning symbols across textbooks and papers.*
*一份统一各教材与论文中常见深度学习符号的速查表。*

---

## Main Idea

In deep learning, symbols are not universal. Different authors may use different symbols for the same concept. For example, model parameters may be written as

$$\theta,\quad w,\quad W,\quad \phi,\quad \psi,\quad \omega.$$

Therefore, the most important habit is not memorizing symbols, but asking:

$$\text{What object does this symbol represent?}$$

---

## Common Deep Learning Symbols

| Symbol | Often Means | Plain English Meaning | Common Alternatives |
|:---:|---|---|---|
| $\theta$ | Model parameters | All learnable parameters of the model | $w, W, \phi, \psi, \omega$ |
| $w$ | Weight | A learnable scalar parameter | $\theta, \beta$ |
| $W$ | Weight matrix | A matrix of learnable weights | $\Theta, A$ |
| $b$ | Bias | Learnable offset term | $c, \beta_0$ |
| $\phi$ | Encoder or feature extractor parameters | Parameters of one part of the model | $\theta_e, \psi$ |
| $\psi$ | Auxiliary model parameters | Often used for critic, value network, or another module | $\theta_v, \omega$ |
| $\omega$ | Generic parameters | Another symbol for learnable parameters | $\theta, w$ |
| $x$ | Input | Data fed into the model | $X, \mathbf{x}$ |
| $y$ | Target label | Correct answer or ground truth | $t, r$ |
| $\hat{y}$ | Prediction | Model output estimate | $y_{\text{pred}}, \tilde{y}$ |
| $f_\theta$ | Model / neural network | A function parameterized by $\theta$ | $h_\theta, g_\theta, M_\theta$ |
| $p_\theta(y \mid x)$ | Probabilistic model | Model's predicted probability distribution | $P(y \mid x;\theta)$ |
| $\mathcal{L}$ | Loss function | Measures how bad the prediction is | $L, J, \text{cost}$ |
| $J(\theta)$ | Objective function | Function to minimize or maximize | $\mathcal{L}(\theta)$ |
| $\nabla_\theta \mathcal{L}$ | Gradient | Direction showing how the loss changes with parameters | $\text{grad}$ |
| $\eta$ | Learning rate | Step size during training | $\alpha, \text{lr}$ |
| $\alpha$ | Learning rate or coefficient | Often a step size or weighting factor | $\eta, \lambda$ |
| $\lambda$ | Regularization strength | Controls penalty term | weight decay coefficient |
| $\beta$ | Momentum coefficient or parameter | Often used in optimization or statistics | decay rate |
| $z$ | Logit or latent variable | Raw score before activation, or hidden latent factor | latent code |
| $h$ | Hidden representation | Internal feature vector | embedding, hidden state |
| $a$ | Activation or action | In neural nets: activation; in RL: action | $h$ |
| $\sigma$ | Activation function | Nonlinear function | sigmoid, activation |
| $D$ or $\mathcal{D}$ | Dataset | Collection of training examples | training set |
| $n$ or $N$ | Number of samples | Dataset size or sample count | $m$ |
| $B$ | Batch size | Number of examples used per update | mini-batch size |
| $i$ | Index | Refers to the $i$-th example | sample index |
| $t$ | Time step or target | In sequence models: time; in supervised learning: target | step |
| $T$ | Sequence length | Total number of time steps | horizon |
| $d$ | Dimension | Size of input or feature vector | feature dimension |
| $K$ | Number of classes | Total class count | $C$ |
| $C$ | Classes or channels | In classification: classes; in CNNs: channels | $K$ |
| $\mathbb{R}^d$ | Real vector space | A vector with $d$ real-valued entries | shape notation |
| $\mathbb{E}$ | Expectation | Average value under a distribution | mean |
| $\sim$ | Sampled from | Indicates a random variable follows a distribution | drawn from |
| $\propto$ | Proportional to | Equal up to a constant factor | proportional |
| $\arg\max$ | Argument of maximum | Choose the input giving the largest value | best choice |
| $\arg\min$ | Argument of minimum | Choose the input giving the smallest value | optimizer target |
| $\theta^*$ | Optimal parameters | Best parameters after training | $\hat{\theta}$ |
| $\hat{\theta}$ | Estimated parameters | Parameters estimated from data | $\theta^*$ |
| $q_\phi$ | Approximate distribution / encoder | Common in variational inference and VAEs | inference model |
| $p_\theta$ | Generative model / decoder | Model distribution parameterized by $\theta$ | likelihood model |
| $\pi$ | Policy or probability distribution | In RL, the model that chooses actions | $p, q, \mu$ |
| $s$ | State | Situation in reinforcement learning | observation |
| $a$ | Action | Action taken by an agent | control |
| $r$ | Reward | Feedback signal in reinforcement learning | reward value |
| $Q(s,a)$ | Action-value function | Expected return after taking action $a$ in state $s$ | critic |
| $V(s)$ | Value function | Expected return from state $s$ | state value |
| $\gamma$ | Discount factor | Controls importance of future rewards | discount |
| $\tau$ | Trajectory or temperature | In RL: sequence of states/actions; in softmax: temperature | path, temp |
| $\mathcal{N}(\mu,\Sigma)$ | Normal distribution | Gaussian distribution with mean and covariance | Gaussian |
| $\mu$ | Mean | Average or center of a distribution | expectation |
| $\Sigma$ | Covariance matrix | Spread and correlation of variables | covariance |
| $\epsilon$ | Small constant or noise | Used for numerical stability or randomness | noise |
| $D_{\mathrm{KL}}$ | KL divergence | Difference between two distributions | relative entropy |
| CE | Cross entropy | Common classification loss | log loss |
| MSE | Mean squared error | Common regression loss | $\ell_2$ loss |
| NLL | Negative log-likelihood | Loss based on probability model | log loss |
| SGD | Stochastic gradient descent | Basic gradient-based optimizer | optimizer |
| Adam | Adaptive optimizer | Popular optimizer using moving averages | optimizer |

---

## Very Common Patterns

### Model Prediction

$$\hat{y} = f_\theta(x)$$

This means:

$$\text{Input } x \quad \longrightarrow \quad \text{model } f_\theta \quad \longrightarrow \quad \text{prediction } \hat{y}.$$

Here, $\theta$ means the learnable parameters of the model.

### Training Objective

$$\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)$$

This means:

$$\text{Find the best parameters } \theta^* \text{ that minimize the loss } \mathcal{L}(\theta).$$

### Gradient Descent Update

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

This means:

$$\text{new parameters} = \text{old parameters} - \text{learning rate} \times \text{gradient}.$$

### Encoder and Classifier Example

$$z = f_\phi(x), \qquad \hat{y} = g_\theta(z)$$

This usually means:

$$f_\phi = \text{encoder with parameters } \phi,$$

$$g_\theta = \text{classifier or decoder with parameters } \theta.$$

So $\phi$ and $\theta$ are both parameters, but they belong to different parts of the model.

---

## How to Decode Symbols in a Paper

When reading a paper, ask the following questions:

| Question | Common Symbols |
|---|---|
| Is this data? | $x, y, \mathcal{D}, x_i, y_i$ |
| Is this a learnable parameter? | $\theta, w, W, b, \phi, \psi, \omega$ |
| Is this a model or function? | $f, g, h, p_\theta, q_\phi$ |
| Is this a loss or objective? | $L, \mathcal{L}, J$ |
| Is this a hyperparameter? | $\eta, \alpha, \lambda, \beta, \gamma$ |
| Is this a probability distribution? | $p, q, \pi, \mathcal{N}$ |
| Is this reinforcement learning notation? | $s, a, r, \pi, Q, V, \gamma$ |

---

## Key Takeaway

The most important rule is:

$$\boxed{\theta,\ w,\ W,\ \phi,\ \psi,\ \omega \text{ often all mean learnable parameters.}}$$

Authors use different letters mainly to separate different parts of the model. For example:

$$p_\theta(x \mid z), \qquad q_\phi(z \mid x)$$

usually means:

$$p_\theta = \text{decoder or generative model},$$

$$q_\phi = \text{encoder or approximate inference model}.$$

Both $\theta$ and $\phi$ are learnable parameters, but they belong to different networks.

