import csv
import numpy as np
# from shared.transform_obs import transform_obs_v2 as transform_obs
import matplotlib.pyplot as plt
import os

from shared.transform_obs import transform_obs_v2 as transform_obs

# version = 3
path = "behavior_v2/data_v3/"
raw_list = np.sort(os.listdir(path))

participant_list = [raw_list[x] for x in range(len(raw_list)) if raw_list[x].startswith('3')]
max_speed = 120

print(participant_list)

save_path = 'traj_from_data_v2/sub'

for iSub in range(len(participant_list)):
    
    print(iSub,participant_list[iSub])
    
    discretize = False

    data_path = path+participant_list[iSub]
    file_list = []
    file_number = []
    for file in os.listdir(data_path):
        if file.endswith(".csv") and not file.startswith("prac"):
            #print(file,int(file[1]))
            if file[-6] == 'e':
                file_number.append(int(file[1])*100+int(file[-5]))
            else:
                file_number.append(int(file[1])*100+int(file[-6:-4])) 
            file_list.append(os.path.join(data_path, file))
    file_index = np.array(file_number).argsort()

    reward_list = np.zeros(len(file_index))
    action_list = np.array([])
    step_list = []
    crash_loc_list = []

    max_step = 299
    
    sum_crash = 0
    sum_clear = 0
    for iFile in range(len(file_index)):
        speed_list = []
        with open(file_list[file_index[iFile]],'r') as file:
            # print(file)
            csvreader = csv.reader(file)
            next(csvreader)
            rows = []
            for row in csvreader:
                rows.append(row)
        converted_rows = np.array([list(map(float,rows[i])) for i in range(len(rows))])
        step_reward = [converted_rows[0,2]] #first reward
        prev_reward = converted_rows[0,2] #first reward
        for iStep in range(converted_rows.shape[0]):
            if prev_reward != converted_rows[iStep,2]:
                step_reward.append(converted_rows[iStep,2]-prev_reward)
                prev_reward = converted_rows[iStep,2]
        reward_list[iFile] = np.sum(step_reward)
        crash_indicator = np.sum(np.array(step_reward) < -10) # 1 if crashed
        sum_crash += crash_indicator
        
        unique_time = np.unique(converted_rows[:,0])
        obs_by_time = []
        action = []
        for iTime in range(len(unique_time)):
            index = converted_rows[:,0] == unique_time[iTime]
            obs_by_time.append(converted_rows[index,:])
            action.append(converted_rows[index,1][0])

        obs_by_time = np.array(obs_by_time)
        
        ##remove state/actions after crash
        if crash_indicator == 1:
            # print(iFile)
            action = action[:len(step_reward)]
            obs_by_time = obs_by_time[:len(step_reward),:,:]
            unique_time = unique_time[:len(step_reward)]
            
        lane_speed = [20,20,20] #initial speed
        observation = []
        for iTime in range(len(unique_time)):
            state,lane_speed = transform_obs(obs_by_time[iTime,:,3:],lane_speed,discretize,max_speed)
            observation.append(state)
            
            speed_list.append(state[0]*(max_speed-20)+20)
            
        observation = np.array(observation)
        
        # print(state,lane_speed)
        cleared = False
        last_state = observation[-1,:]
        
        step_list.append(len(observation))

        ## cleared stage
        if crash_indicator == 0 and len(observation) < max_step: #cleared stage
            cleared = True
            remaining_steps = max_step - len(observation)
            
            #bonus_max_speed = max(speed_list) #use maximum speed within the stage for bonus reward
            bonus_max_speed = max_speed #use maximum possible speed for bonus reward

            reward_list[iFile] = reward_list[iFile] + (((bonus_max_speed/10)**2)/5 * remaining_steps)
            sum_clear += 1

        if len(observation) > max_step: #out-of-fuel
            observation = observation[:max_step,:]
            action = action[:max_step] 
            
        crash_loc_index = np.zeros(len(observation))
        if crash_indicator == 1:
            crash_loc_index[-1] = 1
            
        action_list = np.concatenate((action_list,np.array(action)))
        crash_loc_list = np.concatenate((crash_loc_list,np.array(crash_loc_index)))

        if iFile == 0:
            state_list = observation
        else:
            state_list = np.concatenate((state_list,np.array(observation)))
    
    print("sum_clear",sum_clear)
    
    dir_path = save_path+str(participant_list[iSub])

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    np.save(dir_path+'/observations_back.npy',state_list)
    np.save(dir_path+'/actions.npy',action_list)
    np.save(dir_path+'/rewards.npy',reward_list)
    np.save(dir_path+'/steps.npy',step_list)
    np.save(dir_path+'/num_crash.npy',sum_crash)
    np.save(dir_path+'/num_clear.npy',sum_clear)
    np.save(dir_path+'/crash_loc.npy',crash_loc_list)