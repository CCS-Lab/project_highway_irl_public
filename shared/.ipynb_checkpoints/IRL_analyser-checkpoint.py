import os
import numpy as np
from shared.get_reward_function import get_reward_function
from shared.normalize_array import normalize_array
from shared.boost_action import boost_action
from shared.boost_action import polish_action as polish
import matplotlib.pyplot as plt

class IRL_analyser(object):
    
    ##default state
    default_model = 'best_acc_model.ckpt'#'best_discrete_model.ckpt'
    # action_boost = False
    # polish_action = False
    # back_distance = False
    
    def __init__(self,setup):
        self.setup = setup
        if not self.setup.from_data:
            self.setup.full_space = False
        print("merged space is not used for dqn results")
        
        try:
            print("old transformation", self.setup.transform_old)
        except:
            self.setup.transform_old = False #old transformation method (max distance:300, min: -20)
            print("use new transformation")
        try:
            self.default_model = self.setup.target_model
            print("target model", self.default_model)
        except:
            print("no target model, use the default one")                
            
    def check_condition(self,args,mute=False):
        try:
            self.action_boost = args.boost_action
            if not mute:
                print("boost actions")
        except:
            self.action_boost = False
            if not mute:
                print("no action_boost parameter")

        try:
            self.polish_action = args.polish_action
            if not mute:
                print("polish actions")
        except:
            self.polish_action = False
            if not mute:
                print("no polish_action parameter")
        try:
            self.back_distance = args.back_distance
        except:
            self.back_distance = False
            if not mute:
                print("no back_distance parameter")
        
    def obtain_full_space(self):
        
        ##merge state spaces of participants
        # if self.setup.full_space:
        print("loading full space across participants")
        for iSub in range(len(self.setup.sub_list)):
            sub_id = 'sub'+str(self.setup.sub_list[iSub])
            # print('sub_id',sub_id,"loaded")
            base_dir = "Highway_"+sub_id+"_"+self.setup.keyword
            args = np.load('trained_models_AIRL/'+base_dir+str(self.setup.starting_number)+'/setup.npy',allow_pickle=True).item()
            if iSub==0: #check condition only for the first sub
                self.check_condition(args)
            
            if self.setup.transform_old:
                expert_observations = np.load(args.expert_traj_dir+'/observations_back_old.npy')
                expert_actions = np.load(args.expert_traj_dir+'/actions_old.npy')                
            else:
                expert_observations = np.load(args.expert_traj_dir+'/observations_back.npy')
                expert_actions = np.load(args.expert_traj_dir+'/actions.npy')

            if self.action_boost:
                lane_list = expert_observations[:,1]
                if args.version == 1:
                    expert_actions = boost_action(expert_actions,5,lane_list,mute=True)
                elif args.version == 2:
                    expert_actions = boost_action(expert_actions,2,lane_list,mute=True)

            if self.polish_action:
                expert_actions = polish(expert_actions,expert_observations,mute=True)

            if iSub == 0:
                full_observations = expert_observations
                full_actions = expert_actions
            else:
                full_observations = np.concatenate((full_observations,expert_observations),axis=0)
                full_actions = np.concatenate((full_actions, expert_actions))
        # full_reward = np.zeros((full_actions.shape[0],self.setup.num_sim))
        # full_policy = np.zeros((full_actions.shape[0],self.setup.num_sim))
        # full_action_prob = np.zeros((full_actions.shape[0],5,self.setup.num_sim))
        return full_observations,full_actions
    
    def load_trajectory(self):
        if self.setup.from_data:
            self.base_dir = "Highway_"+self.setup.target_sub+"_"+self.setup.keyword
        else:
            self.base_dir = "Highway_"+self.setup.keyword
        
        args = np.load('trained_models_AIRL/'+self.base_dir+str(self.setup.starting_number)+'/setup.npy',allow_pickle=True).tolist()

        if self.setup.recent_model:    
            model_file_list = os.listdir(args.savedir + args.envs_1 + '/')
            model_list = [model_file_list[x] for x in range(len(model_file_list)) if '0' in model_file_list[x][:5]]
            model_number = max([int(model_list[x].split('model')[0]) for x in range(len(model_list))])
            self.model_restore = str(model_number) + 'model.ckpt'
        else:
            self.model_restore = self.default_model
        print("irl model:", self.model_restore)

        if not self.setup.from_data:
            if self.setup.max_model:
                self.dqn_model = "dqn_max_"+args.obs_file.split("/")[1]+".pth"
            else:
                self.dqn_model = "dqn_maxiter_"+args.obs_file.split("/")[1]+".pth"
            print("dqn_model:",self.dqn_model)
            
        self.check_condition(args)
        
        print("sample size",args.traj_length,"batch size",args.min_length)
        
        if self.setup.from_data:
            if self.back_distance:
                expert_observations = np.load(args.expert_traj_dir+'/observations_back.npy')
                if self.setup.transform_old:
                    expert_observations = np.load(args.expert_traj_dir+'/observations_back_old.npy')
            else:
                expert_observations = np.load(args.expert_traj_dir+'/observations.npy')
                if self.setup.transform_old:
                    expert_observations = np.load(args.expert_traj_dir+'/observations_old.npy')
            expert_actions = np.load(args.expert_traj_dir+'/actions.npy')
            if self.setup.transform_old:
                expert_actions = np.load(args.expert_traj_dir+'/actions_old.npy')
        else:
            expert_observations = np.load(args.obs_file)
            expert_actions = np.load(args.action_file)

        lane_list = expert_observations[:,1]
        if self.action_boost:
            if args.version == 1:
                expert_actions = boost_action(expert_actions,5,lane_list)
            elif args.version == 2:
                expert_actions = boost_action(expert_actions,2,lane_list)

        if self.polish_action:
            expert_actions = polish(expert_actions,expert_observations)
        
        self.args = args

        return expert_observations, expert_actions
    
    def model_prediction(self,expert_observations,expert_actions):
        
        self.est_reward = np.zeros((expert_actions.shape[0],self.setup.num_sim))
        self.est_policy = np.zeros((expert_actions.shape[0],self.setup.num_sim))
        self.est_action_prob = np.zeros((expert_actions.shape[0],5,self.setup.num_sim))        
        
        for iSim in range(self.setup.num_sim):
            args = np.load('trained_models_AIRL/'+self.base_dir+str(self.setup.starting_number+iSim)+'/setup.npy',allow_pickle=True).tolist()
            
            try:
                self.best_iters = np.load('trained_models_AIRL/'+self.base_dir+str(self.setup.starting_number+iSim)+'/best_iters.npy',allow_pickle=True).tolist()
            except:
                pass
            
            args.gpu_fraction = self.setup.gpu_fraction
            args.model_restore = self.model_restore #model to restore
            
            #### temporary change ######
            # print(self.setup.state_only)
            try:
                args.state_only = self.setup.state_only
                print("state_only", self.setup.state_only)
            except:
                pass
            
            try:
                args.weighted_y = self.setup.weighted_y
                print("weighted_y", self.setup.weighted_y)
            except:
                pass
            #############################
            self.est_reward[:,iSim],self.est_policy[:,iSim],self.est_action_prob[:,:,iSim] = get_reward_function(args, expert_observations, expert_actions)

        #find inf
        self.inf_index = np.where(np.isinf(self.est_reward))[0]
        # expert_observations = np.delete(expert_observations, self.inf_index, axis = 0)
        self.raw_reward = self.est_reward.copy()
        
        self.est_reward = np.delete(self.est_reward, self.inf_index, axis = 0)
        print("deleted", len(self.inf_index), "inf rewards")
        
        #find outliers
        threshold = np.mean(self.est_reward) + 3 * np.std(self.est_reward)
        self.outlier_index = (self.est_reward>threshold) + (self.est_reward<-threshold)
        print(np.sum(self.outlier_index),"outliers detected")
        
        return np.delete(expert_observations, self.inf_index, axis = 0)
    
    def calculate_accuracy(self,expert_actions,num_action):
        
        self.num_action = num_action
        pred_match_list = np.zeros((num_action+1,3))

        action_pred = np.squeeze(np.argmax(self.est_action_prob,axis=1))
        pred_match_list[0,0] = np.mean(expert_actions == action_pred)

        for iAction in range(5):
            pred_match_list[iAction+1,0] = np.mean(expert_actions[expert_actions==iAction]==action_pred[expert_actions==iAction])
            pred_match_list[iAction+1,1] = np.sum(expert_actions==iAction)
            pred_match_list[iAction+1,2] = np.sum(action_pred==iAction)

        self.mean_pred = pred_match_list
        self.normalized_acc = np.mean(self.mean_pred[1:,0])
        
    def draw_prediction(self,mean_pred):
        
        threshold = 0.2
        bar_width = 0.3

        plt.figure(figsize=(18,6))
        plt.subplot(1,2,1)
        plt.bar(range(self.num_action),mean_pred[1:,0],capsize=10)
        plt.xlabel('Action',fontsize = 20)
        plt.ylabel('Prediction accuracy',fontsize = 20)
        plt.plot([-0.6,4.6],[0.2,0.2], "k--")
        plt.xlim([-0.6,4.6])
        plt.ylim([0,1])
        plt.legend(['chance level'],fontsize = 15)
        plt.title('Mean accuracy = '+str(round(mean_pred[0,0],2)),fontsize = 20)

        x = np.arange(self.num_action)
        plt.subplot(1,2,2)
        plt.bar(x-bar_width/2,mean_pred[1:,1],capsize=10,width = bar_width)
        plt.bar(x+bar_width/2,mean_pred[1:,2],capsize=10,width = bar_width)
        plt.xlabel('Action',fontsize = 20)
        plt.ylabel('Action frequencies',fontsize = 20)
        plt.xlim([-0.6,4.6])
        plt.legend(['observed','predicted'],fontsize = 15)
    
    def speed_function(self,expert_observations,max_speed,speed_step,min_speed=20):
        speed_list = np.array([expert_observations[x][0] for x in range(len(expert_observations))])
        # min_speed = 20 # can also be loaded from config
                 
        num_bins=(max_speed-min_speed)/speed_step+1
        speed_bin = np.linspace(0,1,int(num_bins))
        speed_design = np.array(range(int(min_speed),max_speed+1,speed_step))
        speed_interval = speed_bin[1]/2
        
        speed_list[speed_list<-speed_interval] = -0.4 #outliers
        
        for iBin in range(len(speed_bin)):
            index = (speed_list < (speed_bin[iBin] + speed_interval))*(speed_list >= (speed_bin[iBin] - speed_interval))
            speed_list[index] = speed_bin[iBin]
        
        unique_speed = speed_bin
        # unique_speed = np.unique(speed_list)
        # unique_speed= np.delete(unique_speed,unique_speed<0)

        speed_function = np.zeros((len(unique_speed),self.setup.num_sim))
        speed_ci = np.zeros((len(unique_speed),self.setup.num_sim))
        self.normal_speed_function = np.zeros((len(unique_speed),self.setup.num_sim))
        self.normal_speed_ci = np.zeros((len(unique_speed),self.setup.num_sim))

        for iSim in range(self.setup.num_sim):
            for iSpeed in range(len(unique_speed)):
                index = speed_list == unique_speed[iSpeed]
                if self.setup.reward_outlier_removal:
                    index = index * np.squeeze(~self.outlier_index)
                
                speed_function[iSpeed,iSim] = np.mean(self.est_reward[index,iSim])
                speed_ci[iSpeed,iSim] = np.std(self.est_reward[index,iSim])
            
            nan_list = np.isnan(speed_function[:,iSim])
            print("missing data point", nan_list)
            speed_function[nan_list,iSim] = np.min(speed_function[~nan_list,iSim])
            speed_ci[nan_list,iSim] = 0
            
            self.normal_speed_function[:,iSim] = normalize_array(speed_function[:,iSim])
            self.normal_speed_ci[:,iSim] = speed_ci[:,iSim]/(max(speed_function[:,iSim])-min(speed_function[:,iSim]))
            
            # self.speed_function = speed_function

        self.speed_label = [str(speed_design[x]) for x in range(len(speed_design))]
        original_reward = np.power(speed_design,2)#reward function in the highway env

        self.normal_original_reward = normalize_array(original_reward)
        self.speed_bin = speed_bin
        self.unique_speed = unique_speed
        self.speed_list = speed_list
    
    def draw_speed_function(self,individual=True):
        
        #individual = True if you want to draw ci for single-simulation data
        
        if individual:
            ci_score = np.squeeze(self.normal_speed_ci)
        else:
            ci_score = 1.96 * np.std(self.normal_speed_function,axis=1)/np.sqrt(self.setup.num_sim)
         
        mean_score = np.mean(self.normal_speed_function,axis=1)

        x= range(len(self.unique_speed))

        # if not full_space:
        initial_array = np.zeros(len(self.speed_bin)) #unobserved state = 0 reward
        speed_index = [v in self.unique_speed for v in self.speed_bin]
        initial_array[speed_index] = mean_score
        x = self.speed_bin
        mean_score = initial_array.copy()
        initial_array[speed_index] = ci_score
        ci_score = initial_array

        plt.plot(x,mean_score)
        plt.fill_between(x,(mean_score-ci_score), (mean_score+ci_score), color='b', alpha=.1)
        plt.plot(x,self.normal_original_reward)
        plt.xlabel('Speed')
        plt.ylabel('Normalized reward')
        plt.xticks(x,self.speed_label)
        plt.legend(['Recovered reward','True reward'])
    
        return mean_score,ci_score
    
    def distance_function(self,expert_observations):
        
        ##distance_list = np.array([expert_observations[x][5+(expert_observations[x,1]*2).astype(int)] for x in range(len(expert_observations))])
        distance_list = np.array([expert_observations[x][5+np.round(expert_observations[x,1]*2).astype(int)] for x in range(len(expert_observations))])

        self.original_distance = distance_list.copy()

        distance_bin = [-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        # distance_bin = np.linspace(0,1,11)
        distance_interval = (distance_bin[1]-distance_bin[0])/2

        distance_list[distance_list<0.02] = -0.1 #crash distance < 0.02

        for iBin in range(len(distance_bin)):            
            index = (distance_list < (distance_bin[iBin] + distance_interval))*(distance_list >= (distance_bin[iBin] - distance_interval))
            distance_list[index] = distance_bin[iBin]

        unique_distance = np.unique(distance_list)
        # unique_distance = np.delete(unique_distance,unique_distance<0)

        dist_function = np.zeros((len(unique_distance),self.setup.num_sim))
        self.normal_dist_function = np.zeros((len(unique_distance),self.setup.num_sim))
        dist_ci = np.zeros((len(unique_distance),self.setup.num_sim))
        self.normal_dist_ci = np.zeros((len(unique_distance),self.setup.num_sim))

        for iSim in range(self.setup.num_sim):
            for iDistance in range(len(unique_distance)):
                index = distance_list == unique_distance[iDistance]
                if self.setup.reward_outlier_removal:
                    index = index * np.squeeze(~self.outlier_index)
                dist_function[iDistance,iSim] = np.mean(self.est_reward[index,iSim])
                dist_ci[iDistance,iSim] = np.std(self.est_reward[index,iSim])
            self.normal_dist_function[:,iSim] = normalize_array(dist_function[:,iSim])
            self.normal_dist_ci[:,iSim] = dist_ci[:,iSim]/(max(dist_function[:,iSim])-min(dist_function[:,iSim]))
                
        self.distance_bin = distance_bin
        self.distance_list = distance_list
        self.unique_distance = unique_distance
        
    def draw_distance_function(self,individual=True):
        normal_original_reward_d = np.zeros(len(self.unique_distance)) + 1
        normal_original_reward_d[0] = 0

        if individual:
            ci_score = np.squeeze(self.normal_dist_ci)
        else:
            ci_score = 1.96 * np.std(self.normal_dist_function,axis=1)/np.sqrt(self.setup.num_sim)

        mean_score = np.mean(self.normal_dist_function,axis=1)

        x= range(len(self.unique_distance))

        plt.plot(x,mean_score)
        plt.fill_between(x,(mean_score-ci_score), (mean_score+ci_score), color='b', alpha=.1)
        plt.plot(x,normal_original_reward_d)
        plt.xlabel('Distance from the car ahead')
        plt.ylabel('Normalized reward')
        plt.legend(['Recovered reward','True reward'])
        
        return mean_score, ci_score
        
    def conditional_speed_function(self,low_range,high_range):
        ##conditional speed: when the distance is small/large (split half)
        speed_index = [v in self.unique_speed for v in self.speed_bin]
        
        ##### split-half index
        distance_index = [[0,np.round(len(self.unique_distance)/2).astype(int)-1],[np.round(len(self.unique_distance)/2).astype(int),len(self.unique_distance)-1]]
        
        ##### threshold index: 0.5
        exact_index = int(np.argwhere(self.unique_distance == 0.5))
        distance_index = [[0,exact_index-1],[exact_index,len(self.unique_distance)-1]]
        
        ##### threshold index
        low_index = np.sum(self.unique_distance < low_range[0])
        high_index = np.sum(self.unique_distance < high_range[0])
        
        low_index2 = np.sum(self.unique_distance < low_range[1])
        high_index2 = np.sum(self.unique_distance < high_range[1])
        
        distance_index = [[low_index,low_index2-1],[high_index,high_index2-1]]

        reward_c_list = np.zeros((len(self.normal_original_reward),len(distance_index),self.setup.num_sim))
        speed_function_c = np.zeros((len(self.unique_speed),self.setup.num_sim))
        
        ci_c_list = np.zeros((len(self.normal_original_reward),len(distance_index),self.setup.num_sim))
        speed_ci_c = np.zeros((len(self.unique_speed),self.setup.num_sim))

        for distance_iter in range(len(distance_index)):
            target_distance_low = self.unique_distance[distance_index[distance_iter][0]]
            target_distance_high = self.unique_distance[distance_index[distance_iter][1]]
            #reward_function_c = np.zeros(len(unique_speed))

            for iSim in range(self.setup.num_sim):
                for iSpeed in range(len(self.unique_speed)):
                    index = (self.speed_list == self.unique_speed[iSpeed])*(self.distance_list >= target_distance_low)*(self.distance_list <= target_distance_high)
                    # print(np.sum(index))
                    if self.setup.reward_outlier_removal:
                        index = index * np.squeeze(~self.outlier_index)
                    speed_function_c[iSpeed,iSim] = np.mean(self.est_reward[index,iSim])
                    speed_ci_c[iSpeed,iSim] = np.std(self.est_reward[index,iSim])

                ###attention!!: reward for an unobserved state is considered the minimum reward (0 normalized reward)
                nan_list = np.isnan(speed_function_c[:,iSim])
                print("missing data point", nan_list)
                speed_function_c[nan_list,iSim] = np.min(speed_function_c[~nan_list,iSim])
                speed_ci_c[nan_list,iSim] = 0
                
                reward_c_list[speed_index,distance_iter,iSim] = normalize_array(speed_function_c[:,iSim])
                ci_c_list[speed_index,distance_iter,iSim] = speed_ci_c[:,iSim]/(max(speed_function_c[:,iSim])-min(speed_function_c[:,iSim]))
                # print(speed_function_c[:,iSim])
                
        self.distance_index = distance_index
        self.reward_c_list = reward_c_list
        self.ci_c_list = ci_c_list

        return reward_c_list
    
    
    def draw_conditional_speed(self,individual=True):
        
        # if not full_space:
        x = range(len(self.speed_bin))

        plt.figure(figsize=(18,6))
        for plot_num in range(len(self.distance_index)):
            #plt.figure(figsize=(18,6))
            plt.subplot(1,len(self.distance_index),plot_num+1)
            if individual:
                ci_score = np.squeeze(self.ci_c_list[:,plot_num,:])
            else:
                ci_score = 1.96 * np.std(self.reward_c_list[:,plot_num,:],axis=1)/np.sqrt(self.setup.num_sim)
            mean_score = np.mean(self.reward_c_list[:,plot_num,:],axis=1)
            # print(np.array(mean_score))
            plt.plot(x,mean_score)
            plt.fill_between(x,(mean_score-ci_score), (mean_score+ci_score), color='b', alpha=.1)
            plt.plot(x,self.normal_original_reward)
            plt.title('Condition: normalized distance = ['+str(np.around(self.unique_distance[self.distance_index[plot_num][0]],2))+', '+str(np.around(self.unique_distance[self.distance_index[plot_num][1]],2))+']',fontsize = 15)
            plt.xlabel('Speed',fontsize = 20)
            plt.ylabel('Normalized reward',fontsize = 20)
            plt.xticks(range(0,len(self.unique_speed),2),np.array(self.speed_label)[range(0,len(self.unique_speed),2)])

        plt.legend(['Recovered reward','True reward'],loc='lower center', bbox_to_anchor=(-0.7, -0.25), ncol=2, fontsize = 15)