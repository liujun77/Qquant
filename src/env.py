#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:53:59 2017

@author: junliu
"""

import numpy as np
import pandas as pd


class Stock:
    
    def __init__(self, filename):
        self.data = pd.read_csv(filename, index_col=0)
        self.data.index = pd.DatetimeIndex(self.data.index)
        self.data = self.data.dropna()
        
        self.window = 30
        
        self.training_data = np.zeros((len(self.data)-self.window, self.window), dtype=np.float)
        for i in range(0, len(self.training_data)):
            for j in range(0, 30):
                self.training_data[i,j] = float(self.data['close'].iloc[i+j])
                
        self.stock_index = 0
        
    def reset(self):
        self.stock_index = 0
        return self.training_data[self.stock_index]
    
    def step(self, action): 
        self.stock_index += 1
        action_reward = self.training_data[self.stock_index][self.window-1]- self.training_data[self.stock_index][self.window-2] 
        if (action == 0):
            action_reward = 0
        if (action == 2):
            action_reward = -1 * action_reward

        stock_done = False
        if self.stock_index >= len(self.stock_data)-1:
            stock_done = True
        else:
            stock_done = False
        return self.training_data[self.stock_index], action_reward, stock_done, 0
    
class Dqn:
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        
        #self.state_dim = env.observation_space.shape[0]
        #self.action_dim = env.action_space.n
        
        self.state_dim = day_len
        self.action_dim = 3
        
        
        self.create_Q_network()
        self.create_training_method()
        
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        
        