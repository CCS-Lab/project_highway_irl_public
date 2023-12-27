import numpy as np

## change actions in key prevention periods to the lane-changing actions that triggered key prevention.

def boost_action(expert_action,prevent_threshold,lane_list,mute=False):
    lane_key = np.array([0,2])
    iAction = 0
    action_list = expert_action.copy()
    
    #remove lane changing actions at the boundaries
    
    if mute == False:        
        print("polished action0:",np.sum((lane_list < 0.1)*(action_list==0)), "out of", np.sum(action_list==0))
        print("polished action2:",np.sum((lane_list > 0.9)*(action_list==2)), "out of", np.sum(action_list==2))
    action_list[(lane_list < 0.1)*(action_list==0)]=1 #top lane
    action_list[(lane_list > 0.9)*(action_list==2)]=1 #bottom lane
    
    # for iAction in range(len(action_list)):
    while iAction < len(action_list):
        if np.sum(lane_key == action_list[iAction]) == 1:
            action_list[iAction+1:np.min([iAction+prevent_threshold, len(action_list)-1])] = action_list[iAction]
            iAction+=prevent_threshold
        else:
            iAction+=1
    return action_list

def polish_action(expert_actions, expert_observations,mute=False):
    
    ##remove redundant speed changing actions
    
    overspeed = (expert_observations[:,0]>np.max(expert_observations[:,0])*0.95)*(expert_actions==3)
    underspeed = (expert_observations[:,0]<=0)*(expert_actions==4)
    
    if mute == False:
        print("polished action3:",np.sum(overspeed),"out of",np.sum(expert_actions==3))
        print("polished action4:",np.sum(underspeed),"out of",np.sum(expert_actions==4))
    
    expert_actions[overspeed] = 1
    expert_actions[underspeed] = 1
    
    return expert_actions

def polish_index(expert_actions,expert_observations,version):
    lane_key = np.array([0,2])
    lane_list = expert_observations[:,1]
    iAction = 0
    
    if version == 1:
        prevent_threshold = 5
    elif version == 2:
        prevent_threshold = 2
        
    print("polished action0:",np.sum((lane_list < 0.1)*(expert_actions==0)), "out of", np.sum(expert_actions==0))
    print("polished action2:",np.sum((lane_list > 0.9)*(expert_actions==2)), "out of", np.sum(expert_actions==2))
    index_0 = (lane_list < 0.1)*(expert_actions==0)
    index_2 = (lane_list > 0.9)*(expert_actions==2)
    index_3 = (expert_observations[:,0]>np.max(expert_observations[:,0])*0.95)*(expert_actions==3)
    index_4 = (expert_observations[:,0]<=0)*(expert_actions==4)    
    print("polished action3:",np.sum(index_3),"out of",np.sum(expert_actions==3))
    print("polished action4:",np.sum(index_4),"out of",np.sum(expert_actions==4))
    
    actions_to_polish = index_0 + index_2 + index_3 + index_4
    
    lane_change_index = np.zeros(len(expert_actions))
    action_list = expert_actions.copy()
    # for iAction in range(len(action_list)):
    while iAction < len(action_list):
        if np.sum(lane_key == action_list[iAction]) == 1:
            lane_change_index[iAction+1:np.min([iAction+prevent_threshold, len(lane_change_index)-1])] = 1
            iAction+=prevent_threshold
        else:
            iAction+=1
    return actions_to_polish,lane_change_index==1