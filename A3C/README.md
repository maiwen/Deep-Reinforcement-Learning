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
t时刻开始到回合结束时，总回报：
                          $R_t=\sum_{k=0}^\infty\gamma^k r_{t+k}$
