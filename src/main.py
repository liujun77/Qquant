#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:37:43 2017

@author: junliu
"""

import env
import agent
import numpy as np
import matplotlib.pyplot as plt

EPISODE = 5000 # Episode limitation
STEP = 3000   #300 # Step limitation in an episode
TEST = 1 # The number of experiment test every 100 episode
batch_size = 32



# initialize OpenAI Gym env and dqn agent
#env = gym.make(ENV_NAME)
Env = env.Stock("../data/RB.csv", train_steps=STEP) 
Agent = agent.Dqn()

def test():
    state = Env.reset()
    state = np.reshape(state, [1, Agent.state_dim])
    for i in xrange(STEP):
        action = Agent.action(state)
        next_state,reward,done = Env.step(action)
        next_state = np.reshape(next_state, [1, Agent.state_dim])
        state = next_state
        if done:
            print("reward:{}"
                  .format(Env.back_test(isplot=True)))
            break
    

#%%
print 'begin'
for episode in xrange(EPISODE):

    # initialize task
    state = Env.reset()
    state = np.reshape(state, [1, Agent.state_dim])
    # Train
    for step in xrange(STEP):
        action = Agent.egreedy_action(state) # e-greedy action for trai
    
        next_state,reward,done = Env.step(action)
        next_state = np.reshape(next_state, [1, Agent.state_dim])
        
        # Define reward for agent
        Agent.remember(state,action,reward,next_state,done)
        state = next_state
        if done:
            Agent.update_target_model()
            print("step: {}, episode: {}/{}, score: {}, e: {:.2}"
                  .format(step, episode, EPISODE, Env.back_test(), Agent.epsilon))
            break
    Agent.replay(batch_size)
    #print Env.signal
    #raw_input()
#%%
test()


 