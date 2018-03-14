# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:37:20 2018

@author: vincent
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

# 超参数
BATCH_SIZE = 32
LR = 0.01                   # 学习率
EPSILON = 0.9               # 最优选择动作百分比
GAMMA = 0.9                 # 奖励递减参数
TARGET_REPLACE_ITER = 100   # Q 目标网络的更新频率
MEMORY_CAPACITY = 2000      # 记忆库大小
EPISODES = 500
# STEPS = 200
# 环境
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        # 不是图像的像素输入，所以没有卷积层而采取了简单的两层全连接层
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.fc2(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, 2 * N_STATES + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr= LR)
        self.loss_function = nn.MSELoss()

    # 根据环境观测值选择动作
    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0))
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    # 存储记忆
    def store_memory(self,s, a, r, s_):
        record = np.hstack((s,[a,r],s_))
        index = self.memory_counter % MEMORY_CAPACITY  # 记忆库满了,新数据覆盖老数据
        self.memory[index, :] = record
        self.memory_counter += 1

    # 学习记忆库中的记忆
    def learn(self):
        # 更新target参数
        if self.learn_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_counter += 1
        # 随机批量提取数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_function(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('clollecting experience...\n')
for i_episode in range(EPISODES):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)
        s_, r, done, info = env.step(a)

        # specific reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        # restore
        dqn.store_memory(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
















