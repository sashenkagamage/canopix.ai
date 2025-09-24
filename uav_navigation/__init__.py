"""
UAV Navigation Package

A Python implementation of A* pathfinding for autonomous UAV navigation
with GPS and LiDAR/SLAM integration.

Key Features:
- 3D A* pathfinding optimized for UAVs
- GPS coordinate conversion utilities
- LiDAR/SLAM integration for real-time obstacle avoidance
- Multiple heuristic functions
- Safety features and collision avoidance
"""

from .grid_node import GridNode, NodeState
from .astar_uav import AStarUAV, SearchConfig
from .occupancy_map import (
    OccupancyMapInterface,
    GPSCoordinateConverter, 
    GridOccupancyMap,
    MockOccupancyMap
)

__version__ = "1.0.0"
__author__ = "UAV Navigation Team"

__all__ = [
    'GridNode',
    'NodeState', 
    'AStarUAV',
    'SearchConfig',
    'OccupancyMapInterface',
    'GPSCoordinateConverter',
    'GridOccupancyMap', 
    'MockOccupancyMap'
]
