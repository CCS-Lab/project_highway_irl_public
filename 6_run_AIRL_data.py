import argparse
import os

from AIRL.AIRL_base_data import AIRL_base
from shared.argparser import argparser
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '3' #use the second GPU

def run_AIRL(args,sub_id,base_name):

    args.gpu_fraction = 1 #0.19
    
    args.n_feature = 11
    
    ##hyperparameters for loading trajectory data
    args.envs_base = 'Highway' #load data from this directory

    args.expert_traj_dir = 'traj_from_data_v2/' + sub_id
    print("load trajectory from",args.expert_traj_dir)
    expert_actions = np.load(args.expert_traj_dir+'/actions.npy')
    args.full_traj_length = len(expert_actions) #length of trajectory data
    print("loaded trajectory size",len(expert_actions))
    reward_data = np.load(args.expert_traj_dir+'/rewards.npy')
    expert_reward = np.mean(reward_data)
    
    data_length = len(expert_actions) - (len(expert_actions)%100) #drop last two digits for parallel sampling
    
    if args.restrict_sample:
        data_length = args.sample_size
        
    print(data_length,"original samples used")
        
    ##hyperparameters that determine the size of the trajectory actually used in the AIRL
    args.traj_length = data_length #number of steps in the expert trajectory
    
    args.split = 5
    
    if args.cv:
        args.cv_length = int(args.traj_length/args.fold)
        print("cv length", args.cv_length, "cv number", args.sim_number)
        base_length = int(args.traj_length) - args.cv_length
    else:        
        base_length = int(args.traj_length)
        
    print(base_length," samples used")
    args.min_length = base_length #agent sample size        
    args.num_expert_dimension = base_length # expert sample size
    args.batch_size_discrim = int(base_length/(args.split)) #500, size of the training batch for the discriminator model
    
    args.random_starting = False
    args.num_parallel_sampler = 2 #number of parallel workers
    
    ##learning rates
    args.lr_policy = 5e-4 #default = 1e-4
    args.lr_value = 5e-4 #default = 1e-4
    args.lr_discrim = 5e-4/args.split #default = 1e-4
    
    ##num epochs
    args.num_epoch_policy = 4 #4 #default = 6
    args.num_epoch_value = 4 #4 #default = 10
    
    ##num_nodes
    args.units_p = [512]*3
    args.units_v = [512]*3
    args.units_d = 512
    
    args.iteration = 1001 #number of IRL iteration
    num_sim = 1 #number of simulations
    
    args.discretize = False
    args.boost_action = True #this was true in state-action pair
    args.polish_action = False
    
    for iSim in range(num_sim):        
        # args.envs_1 ="Highway_"+sub_id+"_take" + str(starting_number+iSim)
        args.envs_1 ="Highway_" + base_name + "_take" + str(args.sim_number+iSim)
        AIRL_base(args,expert_reward)

if __name__ == '__main__':
    
    sim_list = [0] ### [1,2,3, ..., nfold] for cross validation, any numbering for non-cv
    
    for iSim in range(len(sim_list)):
    
        sub_list = np.arange(1) + 303 #subject list
        print(sub_list)

        for iSub in range(len(sub_list)):

            args = argparser()
            warnings.filterwarnings("ignore")
            
            args.state_only = False

            args.prior_model = False
            args.model_restore = 'trained_models_AIRL/Highway_sub999_35qb120_boost_take2/1500model.ckpt'

            sub_id = 'sub'+str(sub_list[iSub])#'sub315'
            base_name = sub_id + "_dqn" #"_35qb120_boost"
            args.config = 'config_35_quick120.npy'
            # args.optim = 'Adam' #Adam or SGD (Default: Adam)

            args.cv = False #cross validation
            args.fold = 4
            args.restrict_sample = False #use a subset of samples
            args.sample_size = 10000 #number of samples to use (if restrict_sample)
            args.sim_number = sim_list[iSim]
            
            print("cv_number",args.sim_number)

            run_AIRL(args,sub_id,base_name)