"""
Occupancy Map interface for UAV navigation
Handles LiDAR/SLAM data integration and GPS coordinate conversion
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class OccupancyMapInterface(ABC):
    """Abstract interface for occupancy maps"""
    
    @abstractmethod
    def is_occupied(self, position: np.ndarray) -> bool:
        """Check if a position is occupied by an obstacle"""
        pass
    
    @abstractmethod
    def get_occupancy_at(self, position: np.ndarray) -> float:
        """Get occupancy probability at position (0.0 = free, 1.0 = occupied)"""
        pass


class GPSCoordinateConverter:
    """
    Converts between GPS coordinates and local ENU (East-North-Up) coordinates
    Essential for integrating GPS waypoints with local SLAM maps
    """
    
    def __init__(self, origin_lat: float, origin_lon: float, origin_alt: float = 0.0):
        """
        Initialize converter with origin point
        
        Args:
            origin_lat: Origin latitude in degrees
            origin_lon: Origin longitude in degrees  
            origin_alt: Origin altitude in meters
        """
        self.origin_lat = np.radians(origin_lat)
        self.origin_lon = np.radians(origin_lon)
        self.origin_alt = origin_alt
        
        # Earth parameters
        self.EARTH_RADIUS = 6378137.0  # WGS84 semi-major axis in meters
        self.EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84 flattening
        
        # Pre-compute values for efficiency
        self.cos_lat0 = np.cos(self.origin_lat)
        self.sin_lat0 = np.sin(self.origin_lat)
    
    def gps_to_local(self, lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
        """
        Convert GPS coordinates to local ENU coordinates
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            
        Returns:
            Local coordinates [East, North, Up] in meters
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Difference from origin
        dlat = lat_rad - self.origin_lat
        dlon = lon_rad - self.origin_lon
        dalt = alt - self.origin_alt
        
        # Convert to local ENU coordinates
        # Simplified approximation for small distances
        east = self.EARTH_RADIUS * dlon * self.cos_lat0
        north = self.EARTH_RADIUS * dlat
        up = dalt
        
        return np.array([east, north, up])
    
    def local_to_gps(self, local_pos: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert local ENU coordinates to GPS coordinates
        
        Args:
            local_pos: Local coordinates [East, North, Up] in meters
            
        Returns:
            GPS coordinates (latitude, longitude, altitude) in degrees and meters
        """
        east, north, up = local_pos
        
        # Convert back to GPS
        dlat = north / self.EARTH_RADIUS
        dlon = east / (self.EARTH_RADIUS * self.cos_lat0)
        dalt = up
        
        lat = np.degrees(self.origin_lat + dlat)
        lon = np.degrees(self.origin_lon + dlon)
        alt = self.origin_alt + dalt
        
        return lat, lon, alt


class GridOccupancyMap(OccupancyMapInterface):
    """
    3D grid-based occupancy map for UAV navigation
    Integrates with LiDAR/SLAM data
    """
    
    def __init__(self, 
                 bounds_min: np.ndarray,
                 bounds_max: np.ndarray, 
                 resolution: float = 0.5,
                 safety_margin: float = 1.0):
        """
        Initialize 3D occupancy grid
        
        Args:
            bounds_min: Minimum bounds [x, y, z] in meters
            bounds_max: Maximum bounds [x, y, z] in meters
            resolution: Grid resolution in meters
            safety_margin: Safety margin around obstacles in meters
        """
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.resolution = resolution
        self.safety_margin = safety_margin
        
        # Calculate grid dimensions
        self.grid_size = np.ceil((bounds_max - bounds_min) / resolution).astype(int)
        
        # Initialize occupancy grid (0.0 = free, 1.0 = occupied)
        self.grid = np.zeros(self.grid_size, dtype=np.float32)
        
        # Track last update time for each cell
        self.update_times = np.zeros(self.grid_size, dtype=np.float64)
        
    def world_to_grid(self, position: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid indices"""
        grid_pos = (position - self.bounds_min) / self.resolution
        return np.floor(grid_pos).astype(int)
    
    def grid_to_world(self, grid_idx: np.ndarray) -> np.ndarray:
        """Convert grid indices to world coordinates"""
        return grid_idx * self.resolution + self.bounds_min
    
    def is_valid_index(self, grid_idx: np.ndarray) -> bool:
        """Check if grid index is within bounds"""
        return (np.all(grid_idx >= 0) and 
                np.all(grid_idx < self.grid_size))
    
    def update_from_lidar(self, 
                         lidar_points: np.ndarray,
                         sensor_position: np.ndarray,
                         timestamp: float = None):
        """
        Update occupancy grid from LiDAR point cloud
        
        Args:
            lidar_points: Point cloud data [N, 3] in world coordinates
            sensor_position: Current sensor position [x, y, z]
            timestamp: Update timestamp (current time if None)
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Mark obstacles from LiDAR points
        for point in lidar_points:
            grid_idx = self.world_to_grid(point)
            if self.is_valid_index(grid_idx):
                self.grid[tuple(grid_idx)] = 1.0
                self.update_times[tuple(grid_idx)] = timestamp
        
        # TODO: Implement ray tracing to mark free space
        # This would trace rays from sensor_position to each lidar_point
        # and mark intermediate cells as free
    
    def apply_safety_margin(self):
        """Dilate obstacles by safety margin"""
        if self.safety_margin <= 0:
            return
            
        margin_cells = int(np.ceil(self.safety_margin / self.resolution))
        
        # Create dilation kernel
        kernel_size = 2 * margin_cells + 1
        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        
        center = margin_cells
        for i in range(kernel_size):
            for j in range(kernel_size):
                for k in range(kernel_size):
                    dist = np.sqrt((i-center)**2 + (j-center)**2 + (k-center)**2)
                    if dist <= margin_cells:
                        kernel[i, j, k] = 1
        
        # Apply dilation (simplified - would use scipy.ndimage in practice)
        original_grid = self.grid.copy()
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    if original_grid[i, j, k] > 0.5:  # If occupied
                        # Dilate around this cell
                        for di in range(-margin_cells, margin_cells + 1):
                            for dj in range(-margin_cells, margin_cells + 1):
                                for dk in range(-margin_cells, margin_cells + 1):
                                    ni, nj, nk = i + di, j + dj, k + dk
                                    if (0 <= ni < self.grid_size[0] and
                                        0 <= nj < self.grid_size[1] and
                                        0 <= nk < self.grid_size[2]):
                                        if kernel[di + margin_cells, dj + margin_cells, dk + margin_cells] > 0:
                                            self.grid[ni, nj, nk] = max(self.grid[ni, nj, nk], 0.8)
    
    def is_occupied(self, position: np.ndarray) -> bool:
        """Check if position is occupied (implements interface)"""
        grid_idx = self.world_to_grid(position)
        
        if not self.is_valid_index(grid_idx):
            return True  # Consider out-of-bounds as occupied
        
        return self.grid[tuple(grid_idx)] > 0.5
    
    def get_occupancy_at(self, position: np.ndarray) -> float:
        """Get occupancy probability at position (implements interface)"""
        grid_idx = self.world_to_grid(position)
        
        if not self.is_valid_index(grid_idx):
            return 1.0  # Out of bounds = occupied
        
        return float(self.grid[tuple(grid_idx)])
    
    def clear_old_obstacles(self, max_age: float, current_time: float = None):
        """Remove obstacles that haven't been observed recently"""
        if current_time is None:
            current_time = time.time()
        
        old_mask = (current_time - self.update_times) > max_age
        self.grid[old_mask] = 0.0


# Simple mock implementation for testing
class MockOccupancyMap(OccupancyMapInterface):
    """Simple mock occupancy map for testing without LiDAR data"""
    
    def __init__(self, obstacles: list = None):
        """
        Args:
            obstacles: List of obstacle definitions [center, size] where
                      center is [x,y,z] and size is [width, height, depth]
        """
        self.obstacles = obstacles if obstacles is not None else []
    
    def add_obstacle(self, center: np.ndarray, size: np.ndarray):
        """Add a box obstacle"""
        self.obstacles.append((center, size))
    
    def is_occupied(self, position: np.ndarray) -> bool:
        """Check if position intersects any obstacle"""
        for center, size in self.obstacles:
            half_size = size / 2
            if (np.all(position >= center - half_size) and 
                np.all(position <= center + half_size)):
                return True
        return False
    
    def get_occupancy_at(self, position: np.ndarray) -> float:
        """Return 1.0 if occupied, 0.0 if free"""
        return 1.0 if self.is_occupied(position) else 0.0
