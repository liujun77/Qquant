#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:37:43 2017

@author: junliu
"""

import env
import agent
import numpy as np

EPISODE = 10000 # Episode limitation
STEP = 10000   #300 # Step limitation in an episode
TEST = 1 # The number of experiment test every 100 episode
batch_size = 100



# initialize OpenAI Gym env and dqn agent
#env = gym.make(ENV_NAME)
Env = env.Stock("../data/RB.csv") 
Agent = agent.Dqn()

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
        if done or step==STEP-1:
            Agent.update_target_model()
            print("step: {}, episode: {}/{}, score: {}, e: {:.2}"
                  .format(step, episode, EPISODE, Env.total_reward, Agent.epsilon))
            break
    Agent.replay(batch_size)
 
    # Test every 100 episodes
    if episode % 1000 == 0:
        total_reward = 0

        for i in xrange(TEST):
            state = Env.reset()
            state = np.reshape(state, [1, Agent.state_dim])
            for j in xrange(STEP):
                #Env.render()
                action = Agent.action(state)   # direct action for test
                state,reward,done = Env.step(action)
                state = np.reshape(state, [1, Agent.state_dim])
                #print reward
                total_reward += reward
                if done:
                    print 'break'
                    break

        ave_reward = total_reward/TEST
        print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
        #print 'episode: ',episode,'Evaluation Average Reward:',Env.total_reward
        Agent.save('model')
        #if ave_reward >= 500:
        #    print '程式結束' 
        #    break

#%%

total_reward = 0

for i in xrange(TEST):
    state = Env.reset()
    state = np.reshape(state, [1, Agent.state_dim])
    for j in xrange(STEP*2):
        #Env.render()
        action = Agent.action(state)   # direct action for test

        state,reward,done = Env.step(action)
        state = np.reshape(state, [1, Agent.state_dim])
        #print reward
        total_reward += reward
        if action<>4:
            print 'id = ', j, 'action = ', action, 'price = ', state[0][-1], 'tot = ', total_reward
        raw_input()
        if done:
            print 'break'
            break

ave_reward = total_reward/TEST
print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
print 'episode: ',episode,'Evaluation Average Reward:',Env.total_reward