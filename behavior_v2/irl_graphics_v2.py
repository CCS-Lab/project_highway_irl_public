import os
from typing import TYPE_CHECKING, Callable, List, Optional
import numpy as np
import pygame
import time

from highway_env.envs.common.action import ActionType, DiscreteMetaAction, ContinuousAction
from highway_env.envs.common.graphics import EnvViewer, ObservationGraphics, EventHandler
#from highway_env.road.graphics import WorldSurface, RoadGraphics
from irl_road_graphics_v2 import WorldSurface, RoadGraphics
#from highway_env.vehicle.graphics import VehicleGraphics
from irl_vehicle_graphics_v2 import VehicleGraphics

if TYPE_CHECKING:
    from highway_env.envs import AbstractEnv
    from highway_env.envs.common.abstract import Action
        

class EnvViewer2(EnvViewer):
    def __init__(self, env: 'AbstractEnv', config: Optional[dict] = None,
                textX:int = 600,
                textY:int = 20) -> None:
        super().__init__(env)
        self.font = pygame.font.Font('freesansbold.ttf', 25)
        self.textX = textX
        self.textY = textY
        self.key_action=1
        self.cur_event=0
        self.lateral_pressed=False
        self.lateral_pressed_time=0
        self.message_duration = 10
        
    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.action_type:
                if (time.time()-self.lateral_pressed_time)>self.env.vehicle.TAU_LATERAL+0.35:
                        self.lateral_pressed=False
                        self.lateral_pressed_time=0
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT and self.env.action_type.longitudinal and self.lateral_pressed==False:
                        self.key_action=self.env.action_type.actions_indexes["FASTER"]
                        self.cur_event=event
                    if event.key == pygame.K_LEFT and self.env.action_type.longitudinal and self.lateral_pressed==False:
                        self.key_action=self.env.action_type.actions_indexes["SLOWER"]
                        self.cur_event=event
                    if event.key == pygame.K_DOWN and self.env.action_type.lateral and self.lateral_pressed==False:
                        self.key_action=self.env.action_type.actions_indexes["LANE_RIGHT"]
                        self.cur_event=event
                        self.lateral_pressed_time=time.time()
                        self.lateral_pressed=True
                    if event.key == pygame.K_UP and self.env.action_type.lateral and self.lateral_pressed==False:
                        self.key_action=self.env.action_type.actions_indexes["LANE_LEFT"]
                        self.cur_event=event
                        self.lateral_pressed_time=time.time()
                        self.lateral_pressed=True

    def display(self,rewards) -> None:
        """Display the road and vehicles on a pygame window."""
        if not self.enabled:
            return

        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)

        if self.vehicle_trajectory:
            VehicleGraphics.display_trajectory(
                self.vehicle_trajectory,
                self.sim_surface,
                offscreen=self.offscreen)

        RoadGraphics.display_road_objects(
            self.env.road,
            self.sim_surface,
            offscreen=self.offscreen
        )

        if self.agent_display:
            self.agent_display(self.agent_surface, self.sim_surface)
            if not self.offscreen:
                if self.config["screen_width"] > self.config["screen_height"]:
                    self.screen.blit(self.agent_surface, (0, self.config["screen_height"]))
                else:
                    self.screen.blit(self.agent_surface, (self.config["screen_width"], 0))

        RoadGraphics.display_traffic(
            self.env.road,
            self.sim_surface,
            simulation_frequency=self.env.config["simulation_frequency"],
            offscreen=self.offscreen)

        ObservationGraphics.display(self.env.observation_type, self.sim_surface)
        
        

        if not self.offscreen:
            self.screen.blit(self.sim_surface, (0, 0))
            if self.env.config["real_time_rendering"]:
                self.clock.tick(self.env.config["simulation_frequency"])
                
            if self.env.config["show_speed"]:
                speed = self.font.render("Speed : " + 
                                         '{number:.{digits}f}'.format(number=self.env.vehicle.speed, digits=0), 
                                         True, (255, 255, 255))
                self.screen.blit(speed, (25, self.textY))
            
            if self.env.config["show_reward"]:               
                if self.env.steps%self.env.config["policy_frequency"] == 0:
                    self.env.score = self.font.render("Score : " + 
                                             '{number:.{digits}f}'.format(number=np.round(rewards), digits=0), 
                                             True, (255, 255, 255))
                self.screen.blit(self.env.score, (self.textX, self.textY))
            if self.env.time_over is not True:
                if self.env.vehicle.crashed == False:
                    self.env.time_limit=self.font.render("Fuel: "+
                                            '{number:.{digits}f}'.format(number= np.ceil(self.env.config["duration"]-
                                                                         self.env.steps/self.env.config["policy_frequency"]), digits=0), True, (255, 255, 255))
                self.screen.blit(self.env.time_limit, (300, self.textY))
            else:
                time_limit=self.font.render("Out of Fuel", True, (255, 0, 0))
                self.screen.blit(time_limit, (300, self.textY))
                
            if (self.env.vehicle.crashed == True) and (self.env.time_over == False):
                cost = self.font.render('{number:.{digits}f}'.format(number=self.env.config["collision_reward"], digits=0),
                                            True, (255, 0, 0))
                self.screen.blit(cost, (self.textX+90, self.textY+30))
#             elif ((self.env.config["duration"]-self.env.steps/self.env.config["policy_frequency"])%1) < 0.6:
#                 print((self.env.config["duration"]-self.env.steps/self.env.config["policy_frequency"])%1)
#                 reward_per_sec = self.font.render('+{number:.{digits}f}'.format(number=1, digits=0), 
#                                          True, (0, 255, 0))
#                 self.screen.blit(reward_per_sec, (self.textX+120, self.textY+30))
            else:
                reward_per_sec = self.font.render('+{number:.{digits}f}/sec'.format(number=np.round(self.env.current_reward * self.env.config["policy_frequency"]), digits=0), 
                                         True, (0, 255, 0))
                self.screen.blit(reward_per_sec, (self.textX+80, self.textY+30))
                
            if len(self.env.road.vehicles) == 1:
                self.message_duration = 0
            if self.message_duration < 8:
                
                if self.message_duration > 0:
                    self.screen.blit(self.font.render("Stage Clear!", True, (0, 255, 0)), (330, self.textY+70))               
                
                    bonus = self.font.render('Bonus Score +{number:.{digits}f}'.format(number=self.env.bonus, digits=0),
                                                True, (0, 225, 0))
                    #print(self.env.bonus)
                    self.screen.blit(bonus, (300, self.textY+100))
                
                self.message_duration+=1
            
            max_reward=self.font.render("High score : "+
                                        '{number:.{digits}f}'.format(number=np.round(self.env.config["max_reward"]), digits=0), True, (255, 255, 255))
            self.screen.blit(max_reward, (250, 300))
                
            pygame.display.flip()

        if self.SAVE_IMAGES and self.directory:
            pygame.image.save(self.sim_surface, str(self.directory / "highway-env_{}.png".format(self.frame)))
            self.frame += 1