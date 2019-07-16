from TRPO_agent import TRPO_agent_new
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import roboschool
import logging
from baselines import logger
from mlp import MlpPolicy_new
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi
import sys
import baselines.common.tf_util as U
import math
import numpy as np
import random
from expert import *
import matplotlib.pyplot as plt
import time
import pandas as pd
import seaborn as sns
from gym import spaces
plt.style.use('seaborn-white')

seed = 1
sess = U.single_threaded_session()
sess.__enter__()
rank = MPI.COMM_WORLD.Get_rank()
if rank != 0:
    logger.set_level(logger.DISABLED)

workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
set_global_seeds(workerseed)


# # Setting up adversarial environment

# In[ ]:


class action_space(object):
    def __init__(self, env):
        self.env = env
        self.high = np.array([ 1,  1,  1,  1,  1])
        self.low = -np.array([ 1,  1,  1,  1,  1])
        self.shape = env.observation_space.shape
    
    def sample(self):
    
        return self.env.observation_space.sample()
        
        
class adversial_env(object):
    def __init__(self):
        # parameter
        self.env = gym.make("RoboschoolInvertedPendulum-v1")
        self.ratio = 0.7
        self.threshold = np.array([ 0.14244403,  0.07706523,  0.00016789,  0.00789366,  0.02395424])
        self.max_turn = 1000
        self.combine_ratio = 0.05
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.env.observation_space.shape[0],))
        self.observation_space = self.env.observation_space
        self.agent = SmallReactivePolicy(self.env.observation_space, self.env.action_space) # declare sample trained agent
        self.obsr = 0
        self.epi_num = 0
        self.total_score = 0
        self.first = True
        self.run_avg = 0
        self.rvg_list = []
        self.score_list = []
        self.epi_list = []
        self.env.metadata
    
    # define reward function
    def reward(self, st):
        return np.abs(st[3])+0.2*np.abs(st[1])-0.08#(np.abs(st[3])-0.00786861)*100
    
    def step(self, a):
        self.epi_num = self.epi_num + 1
        obs = np.clip(a,-1,1)*self.threshold*self.ratio + self.obsr
        ac = self.agent.act(obs)
        self.obsr, r, done, _ = self.env.step(ac)
        #print( np.clip(a,-1,1),np.clip(a,-1,1)*self.ratio)
        
        if self.epi_num >= self.max_turn:
            done = True
        
        if self.first and done:
            self.first = False
            self.run_avg = self.total_score
        
        final_r = self.reward(self.obsr)
        if done and self.epi_num < self.max_turn:
            final_r = 35 # terminal cost 
        
        self.total_score += final_r
        return self.obsr, final_r, done, 0
        
        
    def seed(self, a):
        pass
    
    def reset(self):
        self.obsr = self.env.reset()
        #print(self.total_score)
        self.run_avg = (self.combine_ratio*self.total_score) + (1-self.combine_ratio)*self.run_avg
        #print(self.run_avg)
        #print(self.epi_num)
        
        self.rvg_list.append(self.run_avg)
        self.score_list.append(self.total_score)
        self.epi_list.append(self.epi_num)
        
        self.epi_num = 0
        self.total_score = 0
        return self.obsr
    
env = adversial_env()


# # Setting up agent

# In[ ]:


env.seed(1)
gym.logger.setLevel(logging.WARN)
class pargm(object):
    def __init__(self):
        self.timesteps_per_batch = 15000 # what to train on
        self.max_kl = 0.01
        self.cg_iters = 10
        self.gamma = 0.995
        self.lam =  0.97# advantage estimation
        self.entcoeff=0.0
        self.cg_damping=0.1
        self.vf_stepsize=1e-3
        self.vf_iters =5
        self.max_timesteps = 1e8
        self.max_episodes=0
        self.max_iters=0  # time constraint
        self.callback=None


def policy_fn(name, ob_space, ac_space):
        return MlpPolicy_new(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    
parg = pargm()
agent1 = TRPO_agent_new('pi1', env, policy_fn, parg)


# # Agent training

# In[ ]:


agent1.learn()


# # Evaluation

# In[ ]:


plt.plot(env.score_list[1:])
plt.show()


# In[ ]:


plt.plot(env.rvg_list[1:])
plt.show()


# In[ ]:


plt.plot(env.epi_list[1:])
plt.show()


# In[ ]:


for i in range(10):
    score = 0
    obs = env.reset()
    done = False
    itr = 0
    do = False
    while done == False:   
        a = agent1.action_ev(obs)
        obs, r, done, _ = env.step(a)
        if done:
            do = True

        score += r
    print(score)


# In[ ]:




