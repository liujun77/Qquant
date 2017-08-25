#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:30:26 2017

@author: junliu
"""
from collections import deque
import random
import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

REPLAY_SIZE = 1000 # experience replay buffer size  

class Dqn:
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque(maxlen = REPLAY_SIZE)

        # init some parameters
        self.state_dim = 30
        self.action_dim = 3
        self.learning_rate = 0.001        
        
        self.gamma = 0.999    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        
        self.target_model = self._create_Q_network()
        self.model = self._create_Q_network()
        self.update_target_model()
    
    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)    
        
    def _create_Q_network(self):

        model = Sequential()
        model.add(Dense(60, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model
    
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def remember(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))
                
    def replay(self, batch_size):
        if len(self.replay_buffer)<batch_size:
            return
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.train_on_batch(state, target)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def egreedy_action(self,state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 

    def action(self,state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) 
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)