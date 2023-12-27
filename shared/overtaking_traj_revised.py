import numpy as np

def overtaking_traj_revised(changing_threshold,overtake_distance_threshold,traj_length,after_over_length,only_changing,target_actions,outlier_removal,far_back_threshold,num_sub,yList,obs_list,action_list,forward_filter,crash_index_full,crash_length):

    over_list_full = []
    over_obs_full = []
    over_action_full = []
    lane_changing_freq = np.zeros(num_sub)
    over_freq = np.zeros(num_sub)

    target_lane = 2 #0 = side lanes, 1 = middle lane, 2 = all
    # far_back_threshold = 0.5 #far enough back distance

    for iSub in range(num_sub):
        #print("processing sub",iSub,"data")
        reward_list = yList[iSub]
        target_obs = obs_list[iSub]
        target_action = action_list[iSub]

        lane_list = target_obs[:,1]

        ### lane changing moment
        lane_index = np.squeeze(np.argwhere(((lane_list < 0.05)*(lane_list>-0.05)) + ((lane_list >0.995)*(lane_list<1.05))))
        lane_index2 = np.squeeze(np.argwhere((lane_list < 0.55)*(lane_list>0.45)))

        if only_changing:           
            lane_index = np.squeeze(np.argwhere((lane_list < 0.15)+(lane_list > 0.85))) ##approaching side lane index
            lane_index2 = np.squeeze(np.argwhere((lane_list > 0.35)*(lane_list < 0.65))) ##approaching middle lane index

        lane_side_distance = target_obs[lane_index][:,6] #distance on the side lane
        lane_side_back_distance = target_obs[lane_index][:,9] #back distance on the side lane
        
        # if front_distance:
        #     lane_side_back_distance = target_obs[lane_index][:,6] #distance on the side lane

        overtake_index_side = np.squeeze(np.argwhere((lane_side_back_distance > overtake_distance_threshold[0])*(lane_side_back_distance < overtake_distance_threshold[1])))

        if forward_filter: #only include states that a backside car suddenly appears: more precise definition of overtaking
            
            lane_index_t1 = lane_index-1
            lane_index_t1[lane_index_t1<0]=0 #avoid "-1" state
            lane_side_back_distance_t1 = target_obs[lane_index_t1][:,9]
            
            # if front_distance:
            #     lane_side_back_distance_t1 = target_obs[lane_index_t1][:,6]
            
            prev_dist = lane_side_back_distance_t1[overtake_index_side] #determine previous distance based on full states
            
            # print("match rate",np.mean(prev_dist == lane_side_back_distance[overtake_index_side-1]))
            
            forward_index = prev_dist >far_back_threshold #== 1 #no or far car -> close back car appears
            overtake_index_side = overtake_index_side[forward_index]

        lane_side_back_distance1 = target_obs[lane_index2][:,8] #back distance on the side lane
        lane_side_back_distance2 = target_obs[lane_index2][:,10] #back distance on the side lane

        # if front_distance:
        #     lane_side_back_distance1 = target_obs[lane_index2][:,5] #distance on the side lane
        #     lane_side_back_distance2 = target_obs[lane_index2][:,7] #distance on the side lane
        
        overtake_condition1 = (lane_side_back_distance1 > overtake_distance_threshold[0])*(lane_side_back_distance1 < overtake_distance_threshold[1])
        overtake_condition2 = (lane_side_back_distance2 > overtake_distance_threshold[0])*(lane_side_back_distance2 < overtake_distance_threshold[1])

        if forward_filter: #only include states that a backside car suddenly appears: more precise definition of overtaking           

            overtake_condition1 = np.squeeze(np.argwhere(overtake_condition1))
            overtake_condition2 = np.squeeze(np.argwhere(overtake_condition2))            
            
            lane_index2_t1 = lane_index2-1
            lane_index2_t1[lane_index2_t1<0]=0 #avoid "-1" state

            lane_side_back_distance1_t1 = target_obs[lane_index2_t1][:,8]
            lane_side_back_distance2_t1 = target_obs[lane_index2_t1][:,10]
            
            # if front_distance:
            #     lane_side_back_distance1_t1 = target_obs[lane_index2_t1][:,5]
            #     lane_side_back_distance2_t1 = target_obs[lane_index2_t1][:,7]
            
            prev_dist1 = lane_side_back_distance1_t1[overtake_condition1]
            prev_dist2 = lane_side_back_distance2_t1[overtake_condition2]

            # print("match rate1",np.mean(prev_dist1 == lane_side_back_distance1[overtake_condition1-1]))
            # print("match rate2",np.mean(prev_dist2 == lane_side_back_distance2[overtake_condition2-1]))
            
            overtake_condition1 = overtake_condition1[prev_dist1 >far_back_threshold]
            overtake_condition2 = overtake_condition2[prev_dist2 >far_back_threshold]
            
            # print(np.mean(prev_dist1 >far_back_threshold),np.mean(prev_dist2 >far_back_threshold))
            overtake_index_middle = np.unique(np.concatenate((overtake_condition1,overtake_condition2)))
            
        else:        
            overtake_index_middle = np.squeeze(np.argwhere(overtake_condition1+overtake_condition2))

        if target_lane == 0:
            overtake_index = overtake_index_side
        elif target_lane == 1:
            overtake_index = overtake_index_middle
        else:
            overtake_index = np.concatenate((overtake_index_middle,overtake_index_side))

        over_freq[iSub] = len(overtake_index)

        lane_changing_index = np.squeeze(np.argwhere((target_action==target_actions[0])+(target_action==target_actions[1])))
        lane_changing_freq[iSub] = len(lane_changing_index)

        overtake_index = np.delete(overtake_index,overtake_index<traj_length) + after_over_length

        over_count = 0
        for iOver in range(len(overtake_index)):
            
            #print(iSub, iOver, len(overtake_index))

            ##exclude crash 
            distance_from_crash = min(abs(overtake_index[iOver]-after_over_length-crash_index_full[iSub]))

            if only_changing: # or (other_actions):
                
                #positive distance
                try:
                    distance_from_change = min(abs(overtake_index[iOver]-after_over_length-lane_changing_index[lane_changing_index<=(overtake_index[iOver]-after_over_length)]))
                except:
                    distance_from_change = 999
                    
                #negative distance
                try:
                    distance_from_change_n = max(overtake_index[iOver]-after_over_length-lane_changing_index[lane_changing_index>(overtake_index[iOver]-after_over_length)])
                    # distance_from_change_n = min(abs(overtake_index[iOver]-after_over_length-lane_changing_index[lane_changing_index>(overtake_index[iOver]-after_over_length)]))
                except:
                    distance_from_change_n = 999
                    print("error",overtake_index[iOver]-after_over_length,lane_changing_index[-1])
                
                for i in range(len(changing_threshold)): #multiple filters
                    change_filter_positive = (distance_from_change >= changing_threshold[i][0]) and (distance_from_change <= changing_threshold[i][1])
                    change_filter_negative = (distance_from_change_n >= changing_threshold[i][0]) and (distance_from_change_n <= changing_threshold[i][1])
                    if i == 0:                        
                        change_filter = change_filter_positive + change_filter_negative
                    else:
                        change_filter += (change_filter_positive + change_filter_negative)
                
            else:
                change_filter = True
    
            if (distance_from_crash > crash_length) and change_filter: #successful overtaking
            # if (distance_from_crash < crash_length) and change_filter: #unsuccessful overtaking

                reward_traj = reward_list[overtake_index[iOver]-traj_length+1:overtake_index[iOver]+1]
                obs_traj = target_obs[overtake_index[iOver]-traj_length+1:overtake_index[iOver]+1,:][None,:,:]
                action_traj = target_action[overtake_index[iOver]-traj_length+1:overtake_index[iOver]+1]

                if over_count == 0:
                    reward_traj_list = reward_traj
                    obs_traj_list = obs_traj
                    action_traj_list = action_traj
                    over_count+=1
                    # print(reward_traj)
                else:
                    reward_traj_list = np.vstack((reward_traj_list,reward_traj))
                    obs_traj_list = np.vstack((obs_traj_list,obs_traj))
                    action_traj_list = np.vstack((action_traj_list,action_traj))

        over_list_full.append(reward_traj_list)
        over_obs_full.append(obs_traj_list)
        over_action_full.append(action_traj_list)

    #overtake traj
    mean_traj_list = np.zeros((traj_length,num_sub))

    for iSub in range(num_sub):
        # print(len(over_list_full[iSub].shape),iSub,num_sub)
        if len(over_list_full[iSub].shape) == 1:
            over_list_full[iSub] = over_list_full[iSub].reshape((over_list_full[iSub].shape[0], 1))
        if outlier_removal:
            inf_index = np.sum(over_list_full[iSub]>outlier_thres,axis=1)>0
        else:
            # print(len(over_list_full[iSub].shape))
            inf_index = np.sum(np.isinf(over_list_full[iSub]),axis=1)>0
        target_traj = over_list_full[iSub][~inf_index,:]

        mean_traj_list[:,iSub] = np.mean(target_traj,axis=0)
    return mean_traj_list, over_list_full, lane_changing_freq, over_freq, over_obs_full, over_action_full