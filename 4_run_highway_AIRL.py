import argparse
# import os
from shared.argparser import argparser
from AIRL.AIRL_base import AIRL_base
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = '2' #GPU number

def run_AIRL(args,base_name):
    ##hyperparameters for loading trajectory data
    args.gpu_fraction = 0.19
    args.envs_base = 'Highway' #load data from this directory_AIRL
    
    ##hyperparameters that determine the size of the trajectory used in the AIRL
    args.split = 5
    args.min_length = int(args.traj_length) #agent sample size
    args.num_expert_dimension = int(args.traj_length) # expert sample size
    args.batch_size_discrim = int(args.traj_length/(args.split)) #size of the training batch for the discriminator model
    
    args.random_starting = True
    args.num_parallel_sampler = 10
    
    ##learning rates
    args.lr_policy = 1e-3 #2e-5 #default = 1e-4
    args.lr_value = 1e-3 #2e-5 #default = 1e-4
    args.lr_discrim = 1e-3/2 #default = 1e-4
    
    ##num epochs
    args.num_epoch_policy = 4 #default = 6
    args.num_epoch_value = 4 #default = 10
    
    args.discretize = False
    args.boost_action = False
    args.polish_action = False
    args.n_feature = 11
    
    base_dir = 'dqn_traj/'+base_name
    try:
        filename = 'dqn/setup/'+base_name+'.p'
        with open(filename, 'rb') as file:
            dqn_setup = pickle.load(file)
        print("loaded dqn setup")
        args.config = dqn_setup.config_file
    except:
        print("no saved setup detected")
        args.config = 'config_35_quick80.npy'

    args.obs_file = base_dir+'/back_observations.npy'
    args.action_file = base_dir+'/back_actions.npy'
 
    expert_actions = np.load(args.action_file)#(expert_traj_dir+args.action_file)
    args.full_traj_length = len(expert_actions) #length of trajectory data
    print("loaded trajectory size",len(expert_actions))
    
    ##num_nodes
    args.units_p = [128]*3 #[512]*3
    args.units_v = [128]*3 #[512]*3
    args.units_d = 128 #128
    
    args.iteration = 1501 #number of IRL iteration    
    starting_number = args.sim_number
    num_sim = 1 #number of simulations
        
    for iSim in range(num_sim):
        args.envs_1 ="Highway_"+base_name+"_take" + str(starting_number+iSim) #model name to save
        
        if os.path.exists(base_dir+"/mean_score.npy"):
            dqn_score = np.load(base_dir+"/mean_score.npy")            
            dir_path = "rewards_record_AIRL/"+args.envs_1
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)                
            np.save("rewards_record_AIRL/"+args.envs_1+"/dqn_score.npy",dqn_score)
            
        AIRL_base(args)
        
if __name__ == '__main__':
    args = argparser()
    warnings.filterwarnings("ignore")
    base_name = '35quick120_512' ### dqn model name
    args.traj_length = 9000
    args.sim_number = 2
    run_AIRL(args,base_name)