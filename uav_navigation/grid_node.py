"""
Grid Node class for UAV A* pathfinding
Represents a single cell in the 3D search space
"""

import numpy as np
from enum import Enum
from typing import Optional, Tuple


class NodeState(Enum):
    """Node states for A* algorithm"""
    UNVISITED = 0
    OPENSET = 1
    CLOSEDSET = 2


class GridNode:
    """
    Represents a single node in the 3D grid for A* pathfinding
    
    Attributes:
        index: 3D grid coordinates (i, j, k)
        position: Real-world 3D coordinates (x, y, z)
        g_score: Cost from start to this node
        f_score: Total estimated cost (g_score + heuristic)
        state: Current node state in A* algorithm
        came_from: Parent node for path reconstruction
        rounds: Search round identifier for memory optimization
    """
    
    def __init__(self, index: Optional[Tuple[int, int, int]] = None):
        self.index = index if index is not None else (0, 0, 0)
        self.position = np.array([0.0, 0.0, 0.0])
        
        # A* algorithm variables
        self.g_score = float('inf')
        self.f_score = float('inf')
        self.state = NodeState.UNVISITED
        self.came_from: Optional['GridNode'] = None
        
        # Memory optimization
        self.rounds = 0
    
    def reset(self, rounds: int):
        """Reset node for new search iteration"""
        self.g_score = float('inf')
        self.f_score = float('inf')
        self.state = NodeState.UNVISITED
        self.came_from = None
        self.rounds = rounds
    
    def __lt__(self, other):
        """Comparison for priority queue (lower f_score has higher priority)"""
        return self.f_score < other.f_score
    
    def __eq__(self, other):
        """Equality comparison based on grid index"""
        if not isinstance(other, GridNode):
            return False
        return self.index == other.index
    
    def __hash__(self):
        """Hash function for using nodes in sets/dictionaries"""
        return hash(self.index)
    
    def __repr__(self):
        return f"GridNode(index={self.index}, g={self.g_score:.2f}, f={self.f_score:.2f}, state={self.state.name})"
