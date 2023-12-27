from typing import List, Tuple, Union

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env.vehicle.behavior import LinearVehicle
# from highway_env.vehicle.controller import MDPVehicle
from irl_control import MDPVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

class SpeedOnlyVehicle(LinearVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 20  # [m/s]
    SPEED_MAX: float = 40  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 data: dict = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer)
        self.data = data if data is not None else {}
        self.collecting_data = True
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.
        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.
        :param action: a high-level action
        """
        if self.crashed:
            return
        action = {}
        speed_change_prob=self.road.rand_prob[self.lane_index[2]]
        if speed_change_prob > 0.9:
            self.speed = self.speed + 5/3
            self.speed = min(self.speed,self.SPEED_MAX)
        elif speed_change_prob < 0.1:
            self.speed = self.speed - 5/3
            self.speed = max(self.speed,self.SPEED_MIN)
        
        action['acceleration']=0
        action['steering']=0
        Vehicle.act(self, action)
        

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed
        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.
        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    

class IRLVehicle(MDPVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_MIN: float = 20  # [m/s]
    SPEED_MAX: float = 80  # [m/s]

    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None,
                 speed_step: int = 0
#                  max_speed: float = 60
                ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_step=speed_step
        self.target_speed=speed
#         self.SPEED_MAX = max_speed

    def act(self, action: Union[dict, str] = None) -> None:
        if action == "FASTER":
            self.speed_step=10/3#5/3
#             self.target_speed = min(self.target_speed,self.SPEED_MAX)
        elif action == "SLOWER":
            self.speed_step=-10/3#-5/3
#             self.target_speed = max(self.target_speed,self.SPEED_MIN)
        else:
            super().act(action)
            return
        action={'acceleration':0,'steering':0}
        
    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        self.clip_actions()
        delta_f = self.action['steering']
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array([np.cos(self.heading + beta),
                                   np.sin(self.heading + beta)])
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        
        if self.crashed:
            self.speed_step=0
            self.speed -= self.speed * dt
        else:
            self.speed += self.speed_step
            self.speed = np.clip(self.speed,self.SPEED_MIN,self.SPEED_MAX)
        self.on_state_update()
