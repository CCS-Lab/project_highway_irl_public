from datetime import datetime
import gym
import highway_irl
import time
import pandas as pd
import numpy as np
import pygame
import os
#import sys

def run_highway(sub_num,session_num,duration,block_num):
    dir_path = 'data/'+sub_num
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    env = gym.make('IRL-v1')
    env.configure({
        "manual_control": True,
        'duration':60,
        "vehicles_count":45, #35
        "vehicles_density": 1,
        'policy_frequency':5,
        "collision_reward": -200,
        "real_time_rendering": True,
        "car_allocation":[0.33,0.34,0.33],
        "screen_width": 800,  # [px]
        "screen_height": 400,  # [px]
        "initial_lane_id": 1,
        "show_reward": True,
        "seed": block_num*1000,
        "initial_speed": 20.0,
        "max_speed":80,
        "road_length": 10000,
        "observation": {
            "type": "Kinematics",
            "absolute": True,
            "normalize": False,
            "vehicles_count": 10

        }
    })
    env.reset()
    done = False

    #duration = 20 #experiment duration in minutes
    #a=0
    print("seed",env.config["seed"])

    beginning_time=time.time()
    i = 0
    while (time.time() - beginning_time) < duration*60:
    #for i in range(5):
        episode_rewards=0
        one_episode_obs_df=pd.DataFrame({'time':[],'control':[],'rewards':[],'presence':[],'x':[],'y':[],'vx':[],'vy':[]})
        env.configure({
            "seed":i+(block_num*1000),
            #"max_reward":episode_rewards
        })
        while not done:
            env.render()
            obs, reward, done, info = env.step(1)
            current_time=time.time()-beginning_time
            time_column=np.array([[current_time]*np.shape(obs)[0]])
            key_column=np.array([[env.viewer.key_action]*np.shape(obs)[0]])
            env.viewer.key_action=1 
            episode_rewards+=reward
            rewards_column=np.array([[episode_rewards]*np.shape(obs)[0]])
            temp_array=np.concatenate((time_column.T,key_column.T,rewards_column.T, obs), axis=1)
            temp_df=pd.DataFrame(temp_array,columns=['time','control','rewards','presence','x','y','vx','vy'])
            one_episode_obs_df=one_episode_obs_df.append(temp_df)

    #         print(current_time) #while loop is repeated every 1 secon
        one_episode_obs_df.to_csv(dir_path+'/'+session_num+'_episode{}.csv'.format(str(i)),index=False)
        i+=1
        env.configure({
            "seed":i+(block_num*1000),
            #"max_reward":episode_rewards
        })
        print("seed",env.config["seed"])
        env.reset()
        done=False
        print('new')

    env.close()
    pygame.quit()
    #sys.exit()
    #os._exit(0)