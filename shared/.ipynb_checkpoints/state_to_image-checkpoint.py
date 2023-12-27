import gym
import pygame
import numpy as np
import os
from shared.transform_obs import transform_obs_v2 as transform_obs

import highway_irl_v2
from behavior_v2.irl_graphics_v2 import EnvViewer2

def state_to_image(filename,pos1,pos2,pos3,pos4,heading):
    env = gym.make('IRL-v2')

    # env = gym.make('IRL-v1')
    env.configure({
        "manual_control": False,
        'duration':60,
        "vehicles_count":20,
        'policy_frequency':5,
        "real_time_rendering": False,
        "car_allocation":[0.33,0.34,0.33],
        "screen_width": 800,  # [px]
        "screen_height": 400,  # [px]
        "show_reward": True,
        "initial_lane_id": 1, #0: top, 1:middle, 2:bottom, default: None
        "seed": 1,
        "initial_speed": 20.0,
        "speed_increment":10,
        "max_speed":120,
        "observation": {
            "type": "Kinematics",
            "absolute": True,
            "normalize": False,
            "vehicles_count": 10
        },
        "prevent_threshold": 5
    })
    env.reset()

    os.environ["SDL_VIDEODRIVER"] = "dummy" #dummy video device
    if env.viewer is None:
        env.viewer = EnvViewer2(env)

    lane_speed = [20,20,20] #initial speed

#     obs, reward, done, info = env.step(1)
#     env.render()
    
    env.observation_type.observer_vehicle.position = pos1
    env.observation_type.env.road.vehicles[1].position = pos2
    env.observation_type.env.road.vehicles[2].position = pos3
    env.observation_type.env.road.vehicles[3].position = pos4
    env.observation_type.observer_vehicle.heading = heading #-1~1
    
    env.render()
    img_name = filename
    # pygame.image.save(env.viewer.screen, img_name)
    pygame.image.save(env.viewer.sim_surface, img_name)