import numpy as np

def transform_obs_v2(obs,prev_speed,discretize,max_speed):
    
    def normalize_state(state,max_distance,min_back):
        low = np.array([20,0,20,20,20,0,0,0,0,0,0])
        high = np.array([max_speed,8,40,40,40,max_distance,max_distance,max_distance,
                         min_back,min_back,min_back])
        normalized_state = (state - low)/(high-low)
        return normalized_state
    
    obs_self = obs[0]
    obs_other = obs[1:]
    
    laneList = [0,4,8]
    max_distance = 105 #300 -> 105
    min_back = -46 #-20 -> -46
    
    lane_distance = [max_distance]*3 #list of lane distance (default: max distance)
    #lane_speed = [20,20,20] #default:initial speed
    back_dist = [-20]*3

    for iLane in range(len(laneList)):
        if len(obs_other[obs_other[:,2] == laneList[iLane]]) > 0:
            
            ###### adjust the lane at the moment of crash ###################################
            for iCars in range(obs_other.shape[0]):
                obs_other[iCars,2] = laneList[np.argmin(abs(laneList-obs_other[iCars,2]))]
            #################################################################################
            
            lane_distance[iLane] = min(np.min(obs_other[obs_other[:,2] == laneList[iLane]][:,1])-obs_self[1], max_distance)
            prev_speed[iLane] = (obs_other[obs_other[:,2] == laneList[iLane]][:,3])[0]

            sorted_distance = np.sort(obs_other[obs_other[:,2] == laneList[iLane]][:,1])-obs_self[1]
            negative_distance = sorted_distance[sorted_distance<0]
            positive_distance = sorted_distance[sorted_distance>=0]
            if len(positive_distance)>0:
                lane_distance[iLane] = min(np.min(positive_distance),max_distance)
            else:
                lane_distance[iLane] = max_distance #### apply this to other conditions?
            if len(negative_distance)>0:
                back_dist[iLane] = max(np.max(negative_distance),min_back)
            else:
                back_dist[iLane] = min_back
    
    own_lane = np.clip(obs_self[2],min(laneList),max(laneList))
    
    if discretize:
        #speed
        #speed_bin = np.array([-0.125,0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1])
        speed_bin = np.array(range(15,max_speed+1,5))#np.array([15,20,25,30,35,40,45,50,55,60])
        own_speed = obs_self[3]
        speed_loc = np.argmin(abs(speed_bin-own_speed))
        obs_self[3] = speed_bin[speed_loc]
        
        #lane
        threshold = 0.03*8
        for iLane in range(len(laneList)):
            if (own_lane > (laneList[iLane] - threshold)) * (own_lane < (laneList[iLane] + threshold)):
                own_lane = laneList[iLane]
        
        #lane distance
        lane_distance = np.array(lane_distance)
        lane_distance[lane_distance<(-0.05*max_distance)] = (-0.05*max_distance)

    state = np.concatenate(([obs_self[3],own_lane],prev_speed,lane_distance,back_dist))
            
    state = normalize_state(state,max_distance,min_back)
    
    return state,prev_speed