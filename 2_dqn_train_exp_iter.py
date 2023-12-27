from dqn.agent import Agent
import gym
import numpy as np
from collections import deque
from shared.transform_obs import transform_obs_v2 as transform_obs
import time
import torch
import pickle
import argparse

import highway_irl_v2
from behavior_v2.irl_graphics_v2 import EnvViewer2

import os
os.environ["SDL_VIDEODRIVER"] = "dummy" #dummy video device

discretize = False

n_episodes= 20000
max_t = 1000 #maximum number of steps
eps_start= 0.5 #initial epsilon
eps_end = 0.001 #final epsilon
eps_decay= 0.999 #decay rate

decay_by_action = False #decay eps every action; decay eps every episode otherwise

n_unit = 512 #default = 64
seed = 0 #random seed

n_iter = 1

save_name = '35quick120_512' #model/score file name extension
config_file = 'config_35_quick120.npy' #config file to load from env_configure

config = np.load('env_configure/'+config_file, allow_pickle = True).tolist()
print("loaded config:",config_file)

try:
    max_speed = config['max_speed']
    print("loaded max_speed:",max_speed)
except:
    max_speed = 60 #default speed
    print("default max_speed:",max_speed)

## setup file
dqn_setup = argparse.ArgumentParser().parse_args()
dqn_setup.eps_start = eps_start
dqn_setup.eps_decay = eps_decay
dqn_setup.eps_end = eps_end
dqn_setup.max_t = max_t
dqn_setup.n_unit = n_unit
dqn_setup.version = 2
dqn_setup.config_file = config_file
dqn_setup.max_speed = max_speed
dqn_setup.decay_by_action = decay_by_action

## save setup
with open('dqn/setup/'+save_name+'.p', 'wb') as f:
    pickle.dump(dqn_setup, f)

scores = [] # list containing score from each episode
scores_window = deque(maxlen=100) # last 100 scores at the end of the episode
done_score = np.zeros((n_episodes,n_iter))
done_steps = np.zeros((n_episodes,n_iter))
#done_count = 0
change_lane = np.array([0,2])
max_score = 0
max_by_iter = 0

for i_iter in range(n_iter):
    sim_start_time = time.time()
    print("iteration", i_iter, "start")
    
    env = gym.make('IRL-v2')
    env.config=config

    if env.viewer is None:
        env.viewer = EnvViewer2(env)

    agent = Agent(state_size=11,action_size=env.action_space.n,seed=seed,n_unit=n_unit)
    
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        env.configure({
            "seed":i_episode,
        })
        env.reset()
        lane_speed = [20,20,20] #initial speed
        obs = env.observation_type.observe()
        state,lane_speed = transform_obs(obs,lane_speed,discretize,max_speed)
        score = 0
        prevent_key = 0
        for t in range(max_t):
            action = agent.act(state,eps)            
            next_obs,reward,done,_ = env.step(action)
            next_state,lane_speed = transform_obs(next_obs,lane_speed,discretize,max_speed)
            
            agent.step(state,action,reward,next_state,done)

            state = next_state
            score += reward
            
            ## bonus reward condition
            if env.is_bonus:
                done = True
                score += env.bonus
                # print("bonus score",env.bonus)
            
            if done:
                done_score[i_episode-1,i_iter] = score
                done_steps[i_episode-1,i_iter] = t+1
                
                if env.is_bonus:
                    done_steps[i_episode-1,i_iter] = (t+1)*1000 #distinguish cleared episodes
                    # print("steps",(t+1)*1000)
                
                scores_window.append(score) ## save the most recent score
                #done_count+=1
                if i_episode %1000==0:
                    print('\rEpisode {}\tAverage Score {:.2f} \tMax Score {:.2f}'.format(i_episode,np.mean(scores_window),np.max(scores_window)))
                    print('elapsed time(sec)',time.time()-sim_start_time)
                    sim_start_time = time.time()
                break

            scores.append(score) ## save the most recent score
            
            if decay_by_action:
                eps = max(eps*eps_decay,eps_end)## decrease the epsilon every action
        
        if not decay_by_action:
            eps = max(eps*eps_decay,eps_end)## decrease the epsilon every episode
            
        if (i_episode %1000==0) and (np.mean(scores_window) > max_score): #save the best agent
            torch.save(agent.qnetwork_local.state_dict(),'dqn/model/dqn_max_'+save_name+'.pth')
            max_score = np.mean(scores_window)
            print("max score saved")
        
    if score > max_by_iter:
        torch.save(agent.qnetwork_local.state_dict(),'dqn/model/dqn_maxiter_'+save_name+'.pth')
        max_by_iter = score
        print("max iter score saved")

    with open('dqn/score/dqn_real_iter_'+save_name+'.p', 'wb') as f:
        pickle.dump([done_steps,done_score], f)