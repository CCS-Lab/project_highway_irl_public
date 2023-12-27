import numpy as np
import logging
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from highway_env.road.lane import LineType, StraightLane, AbstractLane
from highway_env.vehicle.objects import Landmark
from highway_env.road.road import Road, RoadNetwork

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics, objects

logger = logging.getLogger(__name__)

LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]

class Road2(Road):

    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(self,
                 network: RoadNetwork = None,
                 vehicles: List['kinematics.Vehicle'] = None,
                 road_objects: List['objects.RoadObject'] = None,
                 np_random: np.random.RandomState = None,
                 record_history: bool = False) -> None:
        super().__init__(network,vehicles,road_objects,np_random,record_history)
        self.rand_prob=[0,0,0]
