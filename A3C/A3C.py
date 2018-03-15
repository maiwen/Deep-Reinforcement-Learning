# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:04:40 2018

@author: vincent
"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import math, os

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200

env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 100)
        self.mean = nn.Linear(100, a_dim)
        self.sigma = nn.Linear(100, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mean, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        a1 = F.relu(self.a1(x))
        mean = 2 * F.tanh(self.mean(a1))
        sigma = F.softplus(self.sigma(a1)) + 0.001
        c1 = F.relu(self.c1(x))
        values = self.v(c1)
        return mean, sigma, values

    def choose_action(self, x):
        self.training = False
        mean, sigma, info = self.forward(x)
        m = self.distribution(mean=mean.view(1, ).data, std=sigma.view(1, ).data)
        return  m.sample().numpy()

    def loss_func(self, s, a, v_t):
        self.train()
        mean, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        m = self.distribution(mean= mean, std= sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.std) # maybe has others methods
        exp_v = log_prob*td.detach() + 0.005 * entropy
        a_loss = -exp_v # why?
        total_loss = (a_loss + c_loss).mean()
        return  total_loss

class worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(worker, self).__init__()
        self.name = 'worker%i' % name
        self.g_ep, self.g_ep_r, self.res_queue, self.gnet, self.opt = global_ep, global_ep_r, res_queue, gnet, opt
        self.lnet = Net(N_S, N_A)
        self.env = gym.make('Pendulum-v0').unwrapped

    def run(self):
        pass


        