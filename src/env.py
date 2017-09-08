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
from sklearn import metrics, preprocessing
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt

class Stock:
    
    def __init__(self, filename, train_steps=3000):
        #self.data = pd.read_csv(filename, index_col=0)
        #self.data.index = pd.DatetimeIndex(self.data.index)
        #self.data = self.data.dropna()
        
        self.training_data = None
        self.stock_index = 0
        self.stock_index_end = 0

        data = np.sin(np.arange(200*1.0)/15)
        #noise = 5*np.random.rand(200)
        #data = data+noise
        close = data
        diff = np.diff(data)
        diff = np.insert(diff, 0, 0)

        xdata = np.column_stack((close, diff))
        xdata = np.nan_to_num(xdata)
        scaler = preprocessing.StandardScaler()
        xdata = scaler.fit_transform(xdata)
        self.training_data = xdata
        self.signal = np.zeros(self.training_data.shape[0])
        """
        if os.path.exists('train_data.npy'):
            self.training_data = np.load('train_data.npy')
            print 'data loaded. '
        else:
            close = self.data['close'].values
            scaler = preprocessing.StandardScaler()
            close = scaler.fit_transform(close)
            self.training_data = close;
            for i in xrange(29):
                close = shift(close, 1)
                self.training_data = np.column_stack((self.training_data, close))
            
            self.training_data = self.training_data[30:10030, :]
        """
    def reset(self):
        self.stock_index = 0
        self.stock_index_end = self.training_data.shape[0]
        
        return self.training_data[self.stock_index], 
    
    def step(self, action): 
        
        self.stock_index += 1
        idx = self.stock_index
        
        if action==1:
            self.signal[idx] = 100
        elif (action == 0):
            self.signal[idx] = 0
        elif (action == 2):
            self.signal[idx] = -100
        
        stock_done = False
        if idx==self.stock_index_end-1:
            stock_done = True
        
        action_reward = 0
        i = 0;
        while(self.signal[idx-i]==self.signal[idx] and idx-i>=0):
            i+=1
        action_reward = (self.training_data[idx][0]-self.training_data[idx-i][0])*self.signal[idx]
        
        if(self.signal[idx]==0 and self.signal[idx-1]==0):
            action_reward -= 10
        
        return self.training_data[idx], action_reward, stock_done

    def back_test(self, isplot=False):
        equality = np.zeros(self.training_data.shape[0])
        cash = np.zeros(self.training_data.shape[0])
        for i in xrange(1,self.training_data.shape[0]):
            cash[i] = cash[i-1]- (self.signal[i]-self.signal[i-1])*self.training_data[i-1][0] 
            equality[i] = cash[i] + self.training_data[i][0]*self.signal[i]
        #print cash
        if isplot:
            plt.figure(figsize=(10, 8))
            close = self.training_data[:, 0].reshape(-1);
            plt.plot(close, '-')
            idx = np.arange(200)
            idx = idx[self.signal>0]
            p_long = close[self.signal>0]
            plt.plot(idx, p_long, 'o', label='long')
            idx = np.arange(200)
            idx = idx[self.signal<0]
            p_short = close[self.signal<0]
            plt.plot(idx, p_short, 'o', label='short')
            plt.show()
            
        return equality[-1];