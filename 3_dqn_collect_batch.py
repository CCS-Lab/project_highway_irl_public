import os
import torch
from dqn.agent import Agent
from tqdm import tqdm
import gym
from shared.transform_obs import transform_obs_v2 as transform_obs
import numpy as np
import pickle

import highway_irl_v2
from behavior_v2.irl_graphics_v2 import EnvViewer2

def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False

def run(traj_length,model_file,config_file,back_distance,n_unit):
    
    discretize = False

    env = gym.make('IRL-v2')       
    env.config=np.load('env_configure/'+config_file, allow_pickle = True).tolist()
    
    try:
        max_speed = env.config['max_speed']
        print("loaded max_speed:",max_speed)
    except:
        max_speed = 60 #default speed
        print("default max_speed:",max_speed)
    
    env.configure({
        "seed":0,
    })
    
    # env.reset()
    os.environ["SDL_VIDEODRIVER"] = "dummy" #dummy video device
    if env.viewer is None:
        env.viewer = EnvViewer2(env)
        
    state_size = 11
    
    seed = 0
    agent = Agent(state_size=state_size,action_size=5,seed=seed,n_unit=n_unit)
    agent.qnetwork_local.load_state_dict(torch.load(model_file))
    disable_gradient(agent.qnetwork_local)

    algo = agent.qnetwork_local
    ##########################
    total_return = 0.0
    num_episodes = 0

    env.reset()
    obs = env.observation_type.observe()
    lane_speed = [20,20,20] #initial speed
    state,lane_speed = transform_obs(obs,lane_speed,discretize,max_speed)
    
    t = 0
    episode_return = 0.0
    
    obs_list = []
    action_list = []
    episode_list = []
    
    done = False

    for _ in tqdm(range(1, traj_length + 1)):

        tensor_state = torch.from_numpy(state).float().unsqueeze(0).to('cpu')
        action_values = algo(tensor_state)
        action = np.argmax(action_values.cpu().data.numpy())            
            
        next_obs,reward,done,_ = env.step(action)
        next_state,lane_speed = transform_obs(next_obs,lane_speed,discretize,max_speed)
        episode_return += reward
        
        obs_list.append(state)
        action_list.append(action)
        
        state = next_state
        # print(t,state,episode_return)
        
        #### testing bonus condition
        if env.is_bonus:
            done = True
            episode_return += env.bonus
            
        if done:
            t += 1
            num_episodes += 1
            total_return += episode_return
            episode_list.append(episode_return)
            
            env.configure({
                "seed":t+99999,
            })            
            
            env.reset()
            lane_speed = [20,20,20] #initial speed
            obs = env.observation_type.observe()
            state,lane_speed = transform_obs(obs,lane_speed,discretize,max_speed)
            
            episode_return = 0.0
            
    print('Mean return of the expert is ',total_return / num_episodes)
    print('standard deviation of the return is ',np.std(np.array(episode_list)))
    print("number of episodes",num_episodes)
    return obs_list,action_list,total_return / num_episodes
    ############################

if __name__ == '__main__':
    traj_length = 50000
    
    base_name = '35quick120_512'
    recent_model = False #use the final model instead of the best score model    
    try:
        filename = 'dqn/setup/'+base_name+'.p'
        with open(filename, 'rb') as file:
            dqn_setup = pickle.load(file)
        print("loaded dqn setup")
        config_file = dqn_setup.config_file
        back_distance = dqn_setup.back_distance
        n_unit = dqn_setup.n_unit        
    except:
        print("no saved setup detected")
        config_file = 'config_35_quick120.npy'
        back_distance = True
        n_unit = 512 #default = 128
    
    if not recent_model:
        model_file = 'dqn/model/dqn_max_'+base_name+'.pth'
    else:
        model_file = 'dqn/model/dqn_maxiter_'+base_name+'.pth'
    print("loaded dqn model:",model_file)
    
    save_folder = 'dqn_traj/'+base_name
    
    state,action,mean_score = run(traj_length,model_file,config_file,back_distance,n_unit)
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.save(save_folder+"/back_observations.npy", np.array(state))
    np.save(save_folder+"/back_actions.npy", np.array(action))
    np.save(save_folder+"/mean_score.npy", mean_score)