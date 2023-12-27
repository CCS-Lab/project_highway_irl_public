from typing import Dict, Tuple, Optional
import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.envs.common.graphics import EventHandler
from irl_graphics import EnvViewer2
from irl_highway_road import Road2
from irl_vehicle import SpeedOnlyVehicle, IRLVehicle


class IRLEnvV1(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    rewards=0
    PERCEPTION_DISTANCE = 50000
    initial_crashed = 0
    time_over=False
    
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 3,
            "vehicles_count": 5,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -50,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 8,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 60],
            "offroad_terminal": False,
            "car_allocation":[0.3,0.4,0.3],
            "show_reward": True,
            "show_speed": True,
            "seed": 1,
            "initial_speed": 25,
            "max_reward":0,
            
        })
        return config

    def _reset(self) -> None:
        self.config['max_reward']=np.maximum(self.config['max_reward'],self.rewards)
        self._create_road()
        self._create_vehicles()
        self.initial_crashed = False
        self.initial_crashed = 0
        self.rewards=0
        self.current_reward = 0
        self.time_over = False
        self.score = 0
        self.time_limit = self.config['duration']
        self.extend_display = 0 #extended display counter after crashing or being out of fuel
        self.restart = False

    def _create_road(self) -> None:
        self.seed(self.config["seed"])
        """Create a road composed of straight adjacent lanes."""
        self.road = Road2(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=60,length=4000),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        accumulated_vehicles=0
        other_vehicles_lane=[]
        for i in range(self.config['lanes_count']):
            if i == self.config['lanes_count']-1:
                other_vehicles_lane+=[i]*(self.config['vehicles_count']-accumulated_vehicles)
            else:
                other_per_lane=int(self.config['car_allocation'][i]*self.config['vehicles_count'])
                other_vehicles_lane+=[i]*other_per_lane
                accumulated_vehicles+=other_per_lane
        
        self.np_random.shuffle(other_vehicles_lane)
        self.controlled_vehicles = []
        for others in other_per_controlled:
#             controlled_vehicle = self.action_type.vehicle_class.create_random(
#                 self.road,
#                 speed=self.config["initial_speed"],
#                 lane_id=self.config["initial_lane_id"],
#                 spacing=self.config["ego_spacing"]
#             )
            controlled_vehicle = IRLVehicle.create_random(
                self.road,
                speed=self.config["initial_speed"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            controlled_vehicle.enable_lane_change=True
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)
            
            vehicle = SpeedOnlyVehicle.create_random(self.road,speed= 20, spacing=5,lane_id=other_vehicles_lane[0])
            vehicle.enable_lane_change=False
            self.road.vehicles.append(vehicle)
                
            for v_lane_id in other_vehicles_lane[1:]:
                vehicle = SpeedOnlyVehicle.create_random(self.road,speed= 20, spacing=1 / self.config["vehicles_density"],
                                                            lane_id=v_lane_id)
                vehicle.enable_lane_change=False
                #vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.road.rand_prob=[self.road.np_random.rand(),self.road.np_random.rand(),self.road.np_random.rand()]
        obs, reward, done, info = super().step(action)
        if self.vehicle.crashed:
            self.initial_crashed += 1
        self.rewards+=reward
        self._clear_vehicles()
        self.vehicle.speed_step=0
            
        return obs, reward, done, info

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        if self.viewer.cur_event!=0:
            EventHandler.handle_event(self.action_type, self.viewer.cur_event)
            self.viewer.cur_event=0
        if (self.steps/self.config["policy_frequency"] >= self.config["duration"]) and (self.vehicle.crashed==False):
            self.vehicle.crashed=True
            self.time_over=True
        super()._simulate(action)
    
    def render(self, mode: str = 'human'):
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer2(self)

        self.enable_auto_render = True

        self.viewer.display(self.rewards)

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
        
    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        
        if self.vehicle.crashed and self.initial_crashed == 1:
            reward = self.config["collision_reward"] * self.vehicle.crashed #+ self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1)
            if self.time_over:
                reward = 0
        elif self.vehicle.crashed and self.initial_crashed != 1:
            reward = 0
        else:
            #reward = self.config["high_speed_reward"] * (np.clip(scaled_speed, 0, 1) ** 2)
            reward = ((self.vehicle.speed/10)**2)/5
            
#        reward = \
#            + self.config["collision_reward"] * self.vehicle.crashed \
#            + self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1) \
#            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
#        reward = utils.lmap(reward,
#                          [self.config["collision_reward"],
#                           self.config["high_speed_reward"] + self.config["right_lane_reward"]],
#                          [0, 1])
        reward = 0 if not self.vehicle.on_road else reward
            
        self.current_reward = reward
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        n_crashed = len([v for v in self.road.vehicles
                    if v.crashed])
        self.config["initial_lane_id"]=self.vehicle.lane_index[2]
        if self.vehicle.crashed:
            self.config["initial_speed"]=25
        else:
            self.config["initial_speed"]=self.vehicle.speed
        
        #final bonus reward
        if len(self.road.vehicles) == 1:            
            max_reward = ((self.config['reward_speed_range'][1]/10)**2)
            self.bonus = np.round(max_reward*(self.config["duration"] - self.steps/self.config["policy_frequency"]))           
            self.rewards+=self.bonus
            
        #if (self.vehicle.crashed and len(self.road.close_vehicles_to(self.vehicle,distance=80.0))<n_crashed):
        if self.vehicle.crashed:
            self.extend_display +=1 
            
            if self.extend_display > 15:
                self.restart = True     
            
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road) or \
            (len(self.road.vehicles) == 1 and self.vehicle.action['steering']<0.005) # or \
             #self.steps/self.config["policy_frequency"] >= self.config["duration"] or \

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)
    
    def _clear_vehicles(self) -> None:
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle.position[0]>self.vehicle.position[0]-60]