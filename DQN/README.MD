# DQN
![](https://github.com/maiwen/Deep-Reinforcement-Learning/blob/master/DQN/img/Human-level%20control%20through%20deep%20reinforcement%20learning.png)
***
强化学习逐渐引起公众的注意要归功于谷歌DeepMind公司。DeepMind公司最初是由Demis Hassabis, Shane Legg和Mustafa Suleyman于2010年创立的。创始人Hassabis有三重身份：游戏开发者，神经科学家以及人工智能创业者。Hassabis的游戏开发者的身份使人不难理解DeepMind在nature上发表的第一篇论文是以雅达利（atari）游戏为背景的。同时，Hassabis又是国际象棋高手，对棋类很擅长，挑战完简单地atari游戏后再挑战公认的极其难的围棋游戏也很自然，于是就有了AlphaGo和李世石的2016之战和nature上的第二篇论文。一战成名之后，深度强化学习技术再次博得人的眼球。

算法的大体框架是传统强化学习中的Q-learning。Q-learning 方法是异策略时间差分方法。其伪代码如图所示:
![](https://pic1.zhimg.com/80/v2-08ab664521ca4d88c4f30464d234e3b5_hd.jpg)

所谓异策略，是指行动策略（产生数据的策略）和要评估的策略不是一个策略。在图1.1 Q-learning 伪代码中，行动策略（产生数据的策略）是第5行的ε-greedy策略，而要评估和改进的策略是第6行的贪婪策略（每个状态取值函数最大的那个动作）。
所谓时间差分方法，是指利用时间差分目标来更新当前行为值函数。
在Q-learning伪代码中，时间差分目标为![](https://www.zhihu.com/equation?tex=r_t%2B%5Cgamma%5Cmax_aQ%5Cleft%28s_%7Bt%2B1%7D%2Ca%5Cright%29)

DQN对Q-learning的修改主要体现在以下三个方面：
* DQN利用深度卷积神经网络逼近值函数
![](https://github.com/maiwen/Deep-Reinforcement-Learning/blob/master/DQN/img/Human-level%20control%20through%20deep%20reinforcement%20learning%20(1).png)
利用神经网络逼近值函数的做法在强化学习领域早就存在了，可以追溯到上个世界90年代。那时，学者们发现利用神经网络，尤其是深度神经网络去逼近值函数这件事不太靠谱，因为常常出现不稳定不收敛的情况，所以这个方向一直没有突破，直到DeepMind出现。

我们可能会问，DeepMind到底做了什么？

别忘了DeepMind的创始人Hassabis是神经科学的博士。2005年Hassabis就想如何利用人的学习过程提升游戏的智能水平，所以他去伦敦大学开始读认知神经科学的博士，很快他便做出了很突出的成就，在《science》,《nature》等顶级期刊狂发论文。他的研究方向是海马体。为什么选海马体？什么是海马体？

海马体是人大脑中主要负责记忆和学习的部分，Hassabis学认知神经科学的目的是为了提升机器的智能选海马体作为研究方向很自然。

现在我们就可以回答，DeepMind到底做了什么？

他们将认识神经科学的成果应用到了深度神经网络的训练之中！
* DQN利用了经验回放对强化学习的学习过程进行训练
人在睡觉的时候，海马体会把一天的记忆重放给大脑皮层。利用这个启发机制，DeepMind团队的研究人员构造了一种神经网络的训练方法：经验回放。

通过经验回放为什么可以令神经网络的训练收敛且稳定？

原因是：对神经网络进行训练时，存在的假设是独立同分布。而通过强化学习采集到的数据之间存在着关联性，利用这些数据进行顺序训练，神经网络当然不稳定。经验回放可以**打破数据间的关联**。具体是这么做的：![](https://pic3.zhimg.com/80/v2-f97a7e4542c07c326aa74fcfe1e03360_hd.jpg)
如图，在强化学习过程中，智能体将数据存储到一个数据库中，然后利用均匀随机采样的方法从数据库中抽取数据，然后利用抽取的数据对神经网络进行训练。

这种经验回放的技巧可以**打破数据之间的关联性**，该技巧在2013年的NIPS已经发布了。2015年的nature论文进一步提出了目标网络的概念进一步地减小数据间的关联性。
* DQN独立设置了目标网络来单独处理时间差分算法中的TD偏差。
与表格型的Q-learning算法所不同的是，利用神经网络对值函数进行逼近时，值函数的更新步更新的是参数θ，更新方法是梯度下降法。因此图1.1中第6行值函数更新实际上变成了监督学习的一次更新过程，其梯度下降法为：![](https://www.zhihu.com/equation?tex=%5C%5B%0A%5Ctheta_%7Bt%2B1%7D%3D%5Ctheta_t%2B%5Calpha%5Cleft%5Br%2B%5Cgamma%5Cmax_%7Ba%27%7DQ%5Cleft%28s%27%2Ca%27%3B%5Ctheta%5Cright%29-Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%5Cright%5D%5Cnabla+Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%0A%5C%5D)
我们称计算TD目标时所用的网络为TD网络。以往的神经网络逼近值函数时，计算TD目标的动作值函数所用的网络参数θ，与梯度计算中要逼近的值函数所用的网络参数相同，这样就容易使得数据间存在关联性，训练不稳定。为了解决这个问题，DeepMind提出计算TD目标的网络表示为θ-；计算值函数逼近的网络表示为θ ；用于动作值函数逼近的网络每一步都更新，而用于计算TD目标的网络每个固定的步数更新一次。

因此值函数的更新变为：![](https://www.zhihu.com/equation?tex=%5C%5B%0A%5Ctheta_%7Bt%2B1%7D%3D%5Ctheta_t%2B%5Calpha%5Cleft%5Br%2B%5Cgamma%5Cmax_%7Ba%27%7DQ%5Cleft%28s%27%2Ca%27%3B%5Ctheta%5E-%5Cright%29-Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%5Cright%5D%5Cnabla+Q%5Cleft%28s%2Ca%3B%5Ctheta%5Cright%29%0A%5C%5D)

![](https://github.com/maiwen/Deep-Reinforcement-Learning/blob/master/DQN/img/Human-level%20control%20through%20deep%20reinforcement%20learning%20(2).png)
