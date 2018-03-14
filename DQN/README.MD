# DQN

***
强化学习逐渐引起公众的注意要归功于谷歌DeepMind公司。DeepMind公司最初是由Demis Hassabis, Shane Legg和Mustafa Suleyman于2010年创立的。创始人Hassabis有三重身份：游戏开发者，神经科学家以及人工智能创业者。Hassabis的游戏开发者的身份使人不难理解DeepMind在nature上发表的第一篇论文是以雅达利（atari）游戏为背景的。同时，Hassabis又是国际象棋高手，对棋类很擅长，挑战完简单地atari游戏后再挑战公认的极其难的围棋游戏也很自然，于是就有了AlphaGo和李世石的2016之战和nature上的第二篇论文。一战成名之后，深度强化学习技术再次博得人的眼球。

算法的大体框架是传统强化学习中的Q-learning。Q-learning 方法是异策略时间差分方法。其伪代码如图所示:
![](https://pic1.zhimg.com/80/v2-08ab664521ca4d88c4f30464d234e3b5_hd.jpg)

所谓异策略，是指行动策略（产生数据的策略）和要评估的策略不是一个策略。在图1.1 Q-learning 伪代码中，行动策略（产生数据的策略）是第5行的ε-greedy策略，而要评估和改进的策略是第6行的贪婪策略（每个状态取值函数最大的那个动作）。
所谓时间差分方法，是指利用时间差分目标来更新当前行为值函数。
在Q-learning伪代码中，时间差分目标为![](https://www.zhihu.com/equation?tex=r_t%2B%5Cgamma%5Cmax_aQ%5Cleft%28s_%7Bt%2B1%7D%2Ca%5Cright%29)

DQN对Q-learning的修改主要体现在以下三个方面：
* DQN利用深度卷积神经网络逼近值函数
* DQN利用了经验回放对强化学习的学习过程进行训练
* DQN独立设置了目标网络来单独处理时间差分算法中的TD偏差。
