"""
A* Algorithm implementation for UAV 3D pathfinding
Based on the C++ implementation with Python optimizations
"""

import numpy as np
import heapq
import time
from typing import List, Tuple, Optional, Callable, Set
from dataclasses import dataclass

from .grid_node import GridNode, NodeState


@dataclass
class SearchConfig:
    """Configuration parameters for A* search"""
    step_size: float = 1.0
    time_limit: float = 0.2  # seconds
    pool_size: Tuple[int, int, int] = (100, 100, 50)  # x, y, z grid dimensions
    boundary_margin: int = 1  # cells to avoid at grid boundaries


class AStarUAV:
    """
    3D A* pathfinding algorithm optimized for UAV navigation
    
    Features:
    - 26-directional movement in 3D space
    - Multiple heuristic functions
    - Real-time obstacle avoidance
    - Memory-optimized node management
    - Safety features for UAV navigation
    """
    
    def __init__(self, config: SearchConfig = None):
        self.config = config if config is not None else SearchConfig()
        
        # Grid management
        self.pool_size = np.array(self.config.pool_size)
        self.center_idx = self.pool_size // 2
        self.grid_nodes = {}  # Using dict for sparse representation
        
        # Search state
        self.rounds = 0
        self.open_set = []
        self.grid_path = []
        
        # Coordinate conversion
        self.step_size = self.config.step_size
        self.inv_step_size = 1.0 / self.step_size
        self.center_position = np.array([0.0, 0.0, 0.0])
        
        # Occupancy map (to be set externally)
        self.occupancy_map = None
        
        # 26-directional movement in 3D
        self.directions = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.directions.append((dx, dy, dz))
    
    def set_occupancy_map(self, occupancy_map):
        """Set the occupancy map for obstacle checking"""
        self.occupancy_map = occupancy_map
    
    def coord_to_index(self, position: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid indices"""
        relative_pos = position - self.center_position
        index = np.round(relative_pos * self.inv_step_size).astype(int)
        return index + self.center_idx
    
    def index_to_coord(self, index: np.ndarray) -> np.ndarray:
        """Convert grid indices to world coordinates"""
        relative_idx = index - self.center_idx
        position = relative_idx * self.step_size + self.center_position
        return position
    
    def is_valid_index(self, index: np.ndarray) -> bool:
        """Check if grid index is within bounds"""
        margin = self.config.boundary_margin
        return (margin <= index[0] < self.pool_size[0] - margin and
                margin <= index[1] < self.pool_size[1] - margin and
                margin <= index[2] < self.pool_size[2] - margin)
    
    def check_occupancy(self, position: np.ndarray) -> bool:
        """Check if position is occupied by obstacle"""
        if self.occupancy_map is None:
            return False
        return self.occupancy_map.is_occupied(position)
    
    def get_or_create_node(self, index: Tuple[int, int, int]) -> GridNode:
        """Get existing node or create new one"""
        if index not in self.grid_nodes:
            self.grid_nodes[index] = GridNode(index)
        
        node = self.grid_nodes[index]
        if node.rounds != self.rounds:
            node.reset(self.rounds)
        
        return node
    
    # Heuristic functions
    def diagonal_heuristic(self, node1: GridNode, node2: GridNode) -> float:
        """
        Diagonal heuristic accounting for 3D movement costs
        Matches the C++ getDiagHeu implementation
        """
        dx = abs(node1.index[0] - node2.index[0])
        dy = abs(node1.index[1] - node2.index[1])
        dz = abs(node1.index[2] - node2.index[2])
        
        # Find the minimum for 3D diagonal movement
        diag_3d = min(dx, dy, dz)
        dx -= diag_3d
        dy -= diag_3d
        dz -= diag_3d
        
        h = 0.0
        
        # Cost for 3D diagonal movement
        h += np.sqrt(3.0) * diag_3d
        
        # Handle remaining 2D movement
        if dx == 0:
            h += np.sqrt(2.0) * min(dy, dz) + abs(dy - dz)
        elif dy == 0:
            h += np.sqrt(2.0) * min(dx, dz) + abs(dx - dz)
        elif dz == 0:
            h += np.sqrt(2.0) * min(dx, dy) + abs(dx - dy)
        else:
            # All dimensions have remaining distance
            diag_2d = min(dx, dy, dz)
            h += np.sqrt(2.0) * diag_2d
            remaining = [dx - diag_2d, dy - diag_2d, dz - diag_2d]
            h += sum(remaining)
        
        return h
    
    def manhattan_heuristic(self, node1: GridNode, node2: GridNode) -> float:
        """Manhattan distance heuristic"""
        return abs(node1.index[0] - node2.index[0]) + \
               abs(node1.index[1] - node2.index[1]) + \
               abs(node1.index[2] - node2.index[2])
    
    def euclidean_heuristic(self, node1: GridNode, node2: GridNode) -> float:
        """Euclidean distance heuristic"""
        diff = np.array(node2.index) - np.array(node1.index)
        return np.linalg.norm(diff)
    
    def get_heuristic(self, node1: GridNode, node2: GridNode) -> float:
        """Get heuristic value (using diagonal by default)"""
        return self.diagonal_heuristic(node1, node2)
    
    def get_neighbors(self, current: GridNode) -> List[Tuple[GridNode, float]]:
        """Get valid neighbors and their movement costs"""
        neighbors = []
        
        for dx, dy, dz in self.directions:
            neighbor_idx = (
                current.index[0] + dx,
                current.index[1] + dy,
                current.index[2] + dz
            )
            
            # Check bounds
            if not self.is_valid_index(np.array(neighbor_idx)):
                continue
            
            # Get or create neighbor node
            neighbor = self.get_or_create_node(neighbor_idx)
            neighbor.position = self.index_to_coord(np.array(neighbor_idx))
            
            # Check for obstacles
            if self.check_occupancy(neighbor.position):
                continue
            
            # Calculate movement cost
            movement_cost = np.sqrt(dx*dx + dy*dy + dz*dz)
            neighbors.append((neighbor, movement_cost))
        
        return neighbors
    
    def adjust_start_end_points(self, start_pos: np.ndarray, end_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Adjust start and end points if they're inside obstacles
        Returns adjusted positions and success flag
        """
        adjusted_start = start_pos.copy()
        adjusted_end = end_pos.copy()
        
        # Adjust start point if in obstacle
        if self.check_occupancy(adjusted_start):
            direction = (end_pos - start_pos)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
                for i in range(10):  # Max 10 attempts
                    adjusted_start = adjusted_start + direction * self.step_size
                    if not self.check_occupancy(adjusted_start):
                        break
                else:
                    return start_pos, end_pos, False
        
        # Adjust end point if in obstacle
        if self.check_occupancy(adjusted_end):
            direction = (start_pos - end_pos)
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                
                for i in range(10):  # Max 10 attempts
                    adjusted_end = adjusted_end + direction * self.step_size
                    if not self.check_occupancy(adjusted_end):
                        break
                else:
                    return start_pos, end_pos, False
        
        return adjusted_start, adjusted_end, True
    
    def reconstruct_path(self, current: GridNode) -> List[GridNode]:
        """Reconstruct path from goal to start"""
        path = []
        while current is not None:
            path.append(current)
            current = current.came_from
        return path  # Note: path is in reverse order (goal to start)
    
    def search(self, start_pos: np.ndarray, end_pos: np.ndarray, 
               step_size: float = None) -> Tuple[bool, List[np.ndarray], dict]:
        """
        Perform A* search from start to end position
        
        Args:
            start_pos: Starting position in world coordinates
            end_pos: Goal position in world coordinates
            step_size: Grid resolution (optional, uses config default)
        
        Returns:
            success: Whether path was found
            path: List of waypoints from start to goal
            stats: Search statistics
        """
        start_time = time.time()
        self.rounds += 1
        
        if step_size is not None:
            self.step_size = step_size
            self.inv_step_size = 1.0 / step_size
        
        # Set grid center between start and end
        self.center_position = (start_pos + end_pos) / 2
        
        # Adjust points if they're in obstacles
        adj_start, adj_end, success = self.adjust_start_end_points(start_pos, end_pos)
        if not success:
            return False, [], {"error": "Cannot adjust start/end points"}
        
        # Convert to grid indices
        start_idx = self.coord_to_index(adj_start)
        end_idx = self.coord_to_index(adj_end)
        
        if not self.is_valid_index(start_idx) or not self.is_valid_index(end_idx):
            return False, [], {"error": "Start or end point outside grid bounds"}
        
        # Initialize start and end nodes
        start_node = self.get_or_create_node(tuple(start_idx))
        start_node.position = adj_start
        start_node.g_score = 0
        start_node.f_score = self.get_heuristic(start_node, 
                                               self.get_or_create_node(tuple(end_idx)))
        start_node.state = NodeState.OPENSET
        
        end_node = self.get_or_create_node(tuple(end_idx))
        end_node.position = adj_end
        
        # Initialize open set
        self.open_set = [(start_node.f_score, id(start_node), start_node)]
        heapq.heapify(self.open_set)
        
        num_iterations = 0
        nodes_explored = 0
        
        while self.open_set:
            num_iterations += 1
            
            # Check time limit
            if time.time() - start_time > self.config.time_limit:
                return False, [], {
                    "error": "Time limit exceeded",
                    "iterations": num_iterations,
                    "nodes_explored": nodes_explored,
                    "time": time.time() - start_time
                }
            
            # Get node with lowest f_score
            _, _, current = heapq.heappop(self.open_set)
            
            # Skip if already processed
            if current.state == NodeState.CLOSEDSET:
                continue
            
            # Check if reached goal
            if current.index == end_node.index:
                path_nodes = self.reconstruct_path(current)
                path_coords = [self.index_to_coord(np.array(node.index)) 
                             for node in reversed(path_nodes)]
                
                return True, path_coords, {
                    "iterations": num_iterations,
                    "nodes_explored": nodes_explored,
                    "path_length": len(path_coords),
                    "time": time.time() - start_time
                }
            
            # Move current to closed set
            current.state = NodeState.CLOSEDSET
            nodes_explored += 1
            
            # Explore neighbors
            for neighbor, movement_cost in self.get_neighbors(current):
                if neighbor.state == NodeState.CLOSEDSET:
                    continue
                
                tentative_g = current.g_score + movement_cost
                
                if neighbor.state != NodeState.OPENSET:
                    # New node discovered
                    neighbor.state = NodeState.OPENSET
                    neighbor.came_from = current
                    neighbor.g_score = tentative_g
                    neighbor.f_score = tentative_g + self.get_heuristic(neighbor, end_node)
                    heapq.heappush(self.open_set, (neighbor.f_score, id(neighbor), neighbor))
                    
                elif tentative_g < neighbor.g_score:
                    # Better path found
                    neighbor.came_from = current
                    neighbor.g_score = tentative_g
                    neighbor.f_score = tentative_g + self.get_heuristic(neighbor, end_node)
                    heapq.heappush(self.open_set, (neighbor.f_score, id(neighbor), neighbor))
        
        return False, [], {
            "error": "No path found",
            "iterations": num_iterations,
            "nodes_explored": nodes_explored,
            "time": time.time() - start_time
        }
