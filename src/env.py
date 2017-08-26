#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:53:59 2017

@author: junliu
"""

import numpy as np
import pandas as pd
import os
import random

class Stock:
    
    def __init__(self, filename, train_steps=3000):
        self.data = pd.read_csv(filename, index_col=0)
        self.data.index = pd.DatetimeIndex(self.data.index)
        self.data = self.data.dropna()
        self.total_reward = 0
        self.window = 30
        # len(self.data)-self.window
        self.training_data = None
        self.stock_index = 0
        self.stock_index_end = 0
        self.tran_length = train_steps
        if os.path.exists('train_data.npy'):
            self.training_data = np.load('train_data.npy')
            print 'data loaded. '
        else:
            self.training_data = np.zeros((250000, self.window), dtype=np.float)
            for i in range(0, len(self.training_data)):
                for j in range(0, 30):
                    self.training_data[i,j] = float(self.data['close'].iloc[i+j])
            np.save('train_data.npy', self.training_data)
            print 'data saved. '
        
    def reset(self):
        self.stock_index = random.randrange(250000-self.tran_length)
        self.stock_index_end = self.stock_index + self.tran_length
        self.total_reward = 0
        return self.training_data[self.stock_index]
    
    def step(self, action): 
        self.stock_index += 1
        action_reward = self.training_data[self.stock_index][self.window-1]- self.training_data[self.stock_index][self.window-2] 
        if (action == 0):
            action_reward = 0
        if (action == 2):
            action_reward = -1 * action_reward
        
        
        stock_done = False
#        if self.total_reward<-50:
#            stock_done = True
#            #action_reward = -1000
#        if self.total_reward>0 and self.total_reward<0.7*self.total_reward: 
#            stock_done = True
#            #action_reward = -1000
        if self.stock_index==self.stock_index_end:
            stock_done = True
            
        self.total_reward += action_reward
        return self.training_data[self.stock_index], action_reward, stock_done
    