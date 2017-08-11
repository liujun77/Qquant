#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:30:26 2017

@author: junliu
"""
from collections import deque
import tensorflow as tf
import random
import numpy as np

REPLAY_SIZE = 100 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch    
GAMMA = 0.9
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon

class Dqn:
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque()

        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        
        self.state_dim = 30
        self.action_dim = 3
                
        self.create_Q_network()
        self.create_training_method()
        
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        
    def create_Q_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim, 60])
        b1 = self.bias_variable([60])
        W2 = self.weight_variable([60, self.action_dim])
        b2 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float",[None,self.state_dim])
        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer,W2) + b2

    def create_training_method(self):
        self.action_input = tf.placeholder("float",[None,self.action_dim])
        # one hot presentation
        self.y_input = tf.placeholder("float",[None])
        Q_action = tf.reduce_sum(tf.mul(self.Q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        #self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(0.01,0.1).minimize(self.cost)
        
    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
            
        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()
    
    def train_Q_network(self):
        self.time_step += 1

        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
            })    
        
    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict = {
            self.state_input:[state]})[0]
        ret_action = None
        if random.random() <= self.epsilon:
            ret_action = random.randint(0,self.action_dim - 1)
        else:
            ret_action = np.argmax(Q_value)
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000  #XXX
        return ret_action

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]})[0])


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)
        