#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:37:43 2017

@author: junliu
"""

import env
import agent

ENV_NAME = 'CartPole-v0'
EPISODE = 10000 # Episode limitation
STEP = 1000   #300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode




# initialize OpenAI Gym env and dqn agent
#env = gym.make(ENV_NAME)
Env = env.Stock("../data/RB.csv") 
Agent = agent.Dqn()

print '開始執行'
for episode in xrange(EPISODE):

    # initialize task
    state = Env.reset()

    # Train
    for step in xrange(STEP):
        action = Agent.egreedy_action(state) # e-greedy action for trai
    
        next_state,reward,done,_ = Env.step(action)

        # Define reward for agent
        reward_agent = -1 if done else 0.1
        Agent.perceive(state,action,reward,next_state,done)
        state = next_state
        if done:
            break
 
    # Test every 100 episodes
    if episode % 10 == 0:
        total_reward = 0

        for i in xrange(TEST):
            state = Env.reset()

            for j in xrange(STEP):
                #Env.render()
                action = Agent.action(state)   # direct action for test
                state,reward,done,_ = Env.step(action)
                #print reward
                total_reward += reward
                if done:
                    break

        ave_reward = total_reward/TEST
        print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
        if ave_reward >= 200:
            print '程式結束' 
            break


