import torch
from dqn.agent import Agent
import numpy as np
import time
from shared.normalize_array import normalize_array
import matplotlib.pyplot as plt

class q_analyser(object):
    
    def __init__(self,analyser,n_unit):
        self.analyser = analyser
        self.n_unit = n_unit

    def disable_gradient(self,network):
        for param in network.parameters():
            param.requires_grad = False

    def obtain_q(self,design_space):
        
        state_size = self.analyser.args.n_feature
        num_action = 5

        agent = Agent(state_size=state_size,action_size=num_action,seed=0,n_unit=self.n_unit)
        agent.qnetwork_local.load_state_dict(torch.load('dqn/model/'+self.analyser.dqn_model))
        self.disable_gradient(agent.qnetwork_local)

        action_space = np.zeros(len(design_space))
        q_value_space = np.zeros(len(design_space)) #### for making Q-value functions
        full_q_value = np.zeros((len(design_space),num_action))

        start_time = time.time()  
        for i in range(len(design_space)):  
            # if i%10000 == 0:
            #     print(i,"elapsed time (sec)",time.time() - start_time)
            #     start_time = time.time()
            tensor_state = torch.from_numpy(design_space[i]).float().unsqueeze(0).to("cpu")
            action_values = agent.qnetwork_local(tensor_state)
            action_space[i] = np.argmax(action_values.cpu().data.numpy())
            q_value_space[i] = np.max(action_values.cpu().data.numpy())
            full_q_value[i,:] = action_values.cpu().data.numpy()

        print(i,"elapsed time (sec)",time.time() - start_time)
        
        self.q_values = q_value_space
        self.q_actions = action_space
        self.full_q_value = full_q_value        
        # return q_value_space, action_space, full_q_value
    
    def generate_q_functions(self):
        
        ##speed function
        speed_q = np.zeros(len(self.analyser.unique_speed))
        for iSpeed in range(len(self.analyser.unique_speed)):
            index = self.analyser.speed_list == self.analyser.unique_speed[iSpeed]
            speed_q[iSpeed] = np.mean(self.q_values[index])
        self.normal_speed_q = normalize_array(speed_q)

        ##distance function
        distance_q = np.zeros(len(self.analyser.unique_distance))
        for iDistance in range(len(self.analyser.unique_distance)):
            index = self.analyser.distance_list == self.analyser.unique_distance[iDistance]
            distance_q[iDistance] = np.mean(self.q_values[index])
        self.normal_distance_q = normalize_array(distance_q)
        
        ##conditional speed function
        speed_index = [v in self.analyser.unique_speed for v in self.analyser.speed_bin]
        if not self.analyser.setup.from_data:
            q_function_c = np.zeros((len(self.analyser.unique_speed),1))
            self.q_c_list = np.zeros((len(self.analyser.normal_original_reward),len(self.analyser.distance_index),1))
            for distance_iter in range(len(self.analyser.distance_index)):
                target_distance_low = self.analyser.unique_distance[self.analyser.distance_index[distance_iter][0]]
                target_distance_high = self.analyser.unique_distance[self.analyser.distance_index[distance_iter][1]]
                # print(target_distance_high)

                for iSpeed in range(len(self.analyser.unique_speed)):
                    index = (self.analyser.speed_list == self.analyser.unique_speed[iSpeed])*(self.analyser.distance_list >= target_distance_low)*(self.analyser.distance_list <= target_distance_high)     
                    q_function_c[iSpeed,0] = np.mean(self.q_values[index])
                
                #remove nan values for unobserved states
                nan_list = np.isnan(q_function_c[:,0])
                q_function_c[nan_list,0] = np.min(q_function_c[~nan_list,0])
                self.q_c_list[speed_index,distance_iter,0] = normalize_array(q_function_c[:,0])

    def speed_q_plot(self,mean_speed,ci_speed):
        
        #compare the q function with an IRL function
        x = range(len(self.analyser.unique_speed))
        plt.plot(x,mean_speed)
        plt.fill_between(x,(mean_speed-ci_speed), (mean_speed+ci_speed), color='b', alpha=.1)
        plt.plot(x,self.normal_speed_q)
        plt.xlabel('Speed')
        plt.ylabel('Normalized reward')
        plt.xticks(x,self.analyser.speed_label)
        plt.legend(['Recovered reward','Q function'])
        
    def distance_q_plot(self,mean_distance,ci_distance):
        
        x = range(len(self.analyser.unique_distance))
        plt.figure()
        plt.plot(x,mean_distance)
        plt.fill_between(x,(mean_distance-ci_distance), (mean_distance+ci_distance), color='b', alpha=.1)
        plt.plot(x,self.normal_distance_q)
        plt.xlabel('Distance from the car ahead')
        plt.ylabel('Normalized reward')
        plt.legend(['Recovered reward','Q function'])
        
    def conditional_q_plot(self,individual=True):
        x = range(len(self.analyser.speed_bin))
        plt.figure(figsize=(18,6))
        for plot_num in range(len(self.analyser.distance_index)):
            plt.subplot(1,len(self.analyser.distance_index),plot_num+1)
            if individual:
                ci_score = np.squeeze(self.analyser.ci_c_list[:,plot_num,:])
            else:
                ci_score = 1.96 * np.std(self.analyser.reward_c_list[:,plot_num,:],axis=1)/np.sqrt(self.analyser.setup.num_sim)
            mean_score = np.mean(self.analyser.reward_c_list[:,plot_num,:],axis=1)
            plt.plot(x,mean_score)
            plt.fill_between(x,(mean_score-ci_score), (mean_score+ci_score), color='b', alpha=.1)
            plt.plot(x,self.q_c_list[:,plot_num,:])
            plt.title('Condition: normalized distance = ['
                      +str(np.around(self.analyser.unique_distance[self.analyser.distance_index[plot_num][0]],2))+', '
                      +str(np.around(self.analyser.unique_distance[self.analyser.distance_index[plot_num][1]],2))+']',fontsize = 15)
            plt.xlabel('Speed',fontsize = 20)
            plt.ylabel('Normalized reward',fontsize = 20)
            plt.xticks(range(0,len(self.analyser.unique_speed),2),
                       np.array(self.analyser.speed_label)[range(0,len(self.analyser.unique_speed),2)])

        plt.legend(['Recovered reward','True Q values'],loc='lower center', bbox_to_anchor=(-0.7, -0.25), ncol=2, fontsize = 15)
        
    