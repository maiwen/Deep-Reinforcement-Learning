<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'],['\\(','\\)']]} }); </script> <script type="text/javascript" async src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML"> </script>
# A3C

## 经验回放机制的局限
不同类型的深度神经网络为 DRL 中策略优化任务提供了高效运行的表征形式。 为了缓解传统策略梯度方法与神经网络结合时出现的不稳定性，各类深度策略梯度方法（如 DDPG、 SVG 等）都采用了经验回放机制来消除训练数据间的相关性。

然而经验回放机制存在几个问题：
1. agent 与环境的每次实时交互都需要耗费很多的内存和计算力；
2. 经验回放机制要求 agent 采用离策略（off-policy）方法来进行学习，而off-policy方法只能基于旧策略生成的数据进行更新；
3. 训练依赖于计算能力很强的图形处理器（如GPU）.

## AC框架
AC算法框架被广泛应用于实际强化学习算法中，该框架集成了值函数估计算法和策略梯度算法，是解决实际问题时最常考虑的框架。大家众所周知的alphago便用了AC框架。而且在强化学习领域最受欢迎的A3C算法，DDPG算法，PPO算法等都是AC框架。

### 强化学习方法分类
Value-Based（或Q-Learning）和Policy-Based（或Policy Gradients）是强化学习中最重要的两类方法，区别在于：
* Value-Based是预测某个State下所有Action的期望价值（Q值），之后通过选择最大Q值对应的Action执行策略，适合仅有少量离散取值的Action的环境；
* Policy-Based是直接预测某个State下应该采取的Action，适合高维连续Action的环境，更通用。

根据是否对State的变化进行预测，RL又可以分为model-based和model-free：
* model-based，根据State和采取的Action预测接下来的State，并利用这个信息训练强化学习模型（知道状态的转移概率）；
* model-free，不需对环境状态进行任何预测，也不考虑行动将如何影响环境，直接对策略或Action的期望价值进行预测，计算效率非常高。

因为复杂环境中难以使用model预测接下来的环境状态，所以传统的DRL都是基于model-free。

### 1. Value-Based & model-free
参考DQN

### 2. Policy-Based & model-free
直接将策略参数化: π(a|s,θ)

通过迭代更新 θ，使总回报期望 E[Rt] 梯度上升。 
具体地 
![](http://img.blog.csdn.net/20170613213322872?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzIzNjk0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

①中，π(at|st;θ)表示在 st,θ 的情况下选择动作 at 的概率。概率的对数乘以该动作的总回报 Rt，对 θ 求梯度，以梯度上升的方式更新 θ 。该公式的意义在于，回报越高的动作越努力提高它出现的概率。

但是某些情形下，每个动作的总回报 Rt 都不为负，那么所有的梯度值都大于等于0，此时每个动作出现的概率都会提高，这在很大程度下减缓了学习的速度，而且也会使梯度的方差很大。因此需要对 Rt 使用某种标准化操作来降低梯度的方差。

②具体地，可以让 Rt 减去一个基线 b（baseline），b 通常设为 Rt 的一个期望估计，通过求梯度更新 θ，总回报超过基线的动作的概率会提高，反之则降低，同时还可以降低梯度方差（证明略）。这种方式被叫做行动者-评论家（actor-critic）体系结构，其中策略 π 是行动者，基线 bt 是评论家。

③在实际中，Rt−bt(st) 可使用动作优势函数 Aπ(at,st)=Qπ(at,st)−Vπ(st)代替，因为 Rt 可以视为 Qπ(at,st) 的估计，基线 bt(st) 视为 Vπ(st) 的估计。

### 异步RL框架
论文共实现了四种异步训练的强化学习算法，分别是one-step Q-learning, one-step Sarsa, n-step Q-learning, and advantage actor-critic（A3C）。
$$\nabla_{\theta'}\log{\pi(a_t|s_t;\theta')}A(s_t,a_t;\theta',\theta_v')$$
不同线程的agent，其探索策略不同以保证多样性，不需要经验回放机制，通过各并行agent收集的样本训练降低样本相关性，且学习的速度和线程数大约成线性关系，能适用off-policy、on-policy，离散型、连续型动作。
#### A3C  
![](http://img.blog.csdn.net/20170613220534373?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzIzNjk0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
A3C更新公式有两条，一条梯度上升更新策略 π 参数，如前面介绍actor-critic体系结构， 
![](https://latex.codecogs.com/gif.latex?%5Cnabla_%7B%5Ctheta%27%7D%5Clog%7B%5Cpi%28a_t%7Cs_t%3B%5Ctheta%27%29%7DA%28s_t%2Ca_t%3B%5Ctheta%27%2C%5Ctheta_v%27%29)



