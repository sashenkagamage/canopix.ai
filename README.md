# UAV A* Navigation System

A comprehensive Python implementation of A* pathfinding algorithm specifically designed for autonomous UAV navigation with GPS and LiDAR/SLAM integration.

## Features

- **3D A* Pathfinding**: Optimized for UAV navigation with 26-directional movement
- **Multiple Heuristics**: Diagonal, Manhattan, and Euclidean distance functions
- **GPS Integration**: Seamless conversion between GPS coordinates and local ENU frames
- **LiDAR/SLAM Support**: Real-time obstacle detection and occupancy mapping
- **Safety Features**: Automatic obstacle avoidance and collision checking
- **Real-time Performance**: Optimized for responsive UAV control systems

## Architecture

```
uav_navigation/
├── grid_node.py       # Grid node representation with A* states
├── astar_uav.py       # Core A* algorithm implementation
├── occupancy_map.py   # GPS conversion and obstacle mapping
└── __init__.py        # Package interface
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from uav_navigation import AStarUAV, SearchConfig, MockOccupancyMap

# Configure the planner
config = SearchConfig(
    step_size=1.0,          # 1-meter grid resolution
    time_limit=5.0,         # 5-second search limit
    pool_size=(100, 100, 50) # Search space dimensions
)

# Initialize planner
planner = AStarUAV(config)

# Set up obstacles (replace with real LiDAR data)
occupancy_map = MockOccupancyMap()
occupancy_map.add_obstacle(
    center=np.array([10, 10, 15]), 
    size=np.array([5, 5, 10])
)
planner.set_occupancy_map(occupancy_map)

# Plan path
start_pos = np.array([0, 0, 10])
goal_pos = np.array([20, 20, 15])

success, path, stats = planner.search(start_pos, goal_pos)

if success:
    print(f"Path found with {len(path)} waypoints")
    for waypoint in path:
        print(f"Waypoint: {waypoint}")
```

### GPS Integration

```python
from uav_navigation import GPSCoordinateConverter

# Set GPS origin
converter = GPSCoordinateConverter(
    origin_lat=37.7749,    # San Francisco
    origin_lon=-122.4194,
    origin_alt=100.0
)

# Convert GPS to local coordinates
start_gps = (37.7750, -122.4194, 120.0)
start_local = converter.gps_to_local(*start_gps)

# Plan in local coordinates, then convert back
# ... (planning code) ...

# Convert path back to GPS
gps_waypoints = [converter.local_to_gps(waypoint) for waypoint in path]
```

## Configuration Options

### SearchConfig Parameters

- `step_size`: Grid resolution in meters (default: 1.0)
- `time_limit`: Maximum search time in seconds (default: 0.2)
- `pool_size`: 3D search space dimensions (default: (100, 100, 50))
- `boundary_margin`: Safety margin from grid boundaries (default: 1)

### Heuristic Functions

- **Diagonal Heuristic** (default): Accounts for 3D diagonal movement costs
- **Manhattan Heuristic**: Fast computation, less accurate
- **Euclidean Heuristic**: Straight-line distance, good for open spaces

## Integration with Real Systems

### LiDAR/SLAM Integration

```python
from uav_navigation import GridOccupancyMap

# Create occupancy map from SLAM
occupancy_map = GridOccupancyMap(
    bounds_min=np.array([-50, -50, 0]),
    bounds_max=np.array([50, 50, 30]),
    resolution=0.5,
    safety_margin=2.0
)

# Update from LiDAR data
occupancy_map.update_from_lidar(
    lidar_points=point_cloud,  # Your LiDAR data
    sensor_position=current_position
)

planner.set_occupancy_map(occupancy_map)
```

### Real-time Replanning

```python
def navigation_loop():
    while mission_active:
        # Get current position from GPS/IMU
        current_pos = get_current_position()
        
        # Update occupancy map from LiDAR
        lidar_data = get_lidar_data()
        occupancy_map.update_from_lidar(lidar_data, current_pos)
        
        # Replan if needed
        if need_replanning(current_pos, current_path):
            success, new_path, _ = planner.search(current_pos, goal_pos)
            if success:
                send_path_to_flight_controller(new_path)
        
        time.sleep(0.1)  # 10Hz planning rate
```

## Performance Characteristics

| Configuration | Grid Size | Typical Time | Memory Usage |
|---------------|-----------|--------------|--------------|
| Coarse (2m)   | 50³       | 10-50ms      | ~10MB        |
| Medium (1m)   | 80³       | 50-200ms     | ~50MB        |
| Fine (0.5m)   | 100³      | 100-500ms    | ~100MB       |

## Algorithm Details

### A* Implementation

- **Search Space**: 26-directional movement in 3D (including diagonals)
- **Cost Function**: Distance-based with obstacle penalties
- **Memory Optimization**: Node pooling with round-based reset
- **Safety Features**: Automatic start/goal adjustment if in obstacles

### UAV-Specific Optimizations

1. **3D Movement Costs**: Proper weighting for vertical vs horizontal movement
2. **Safety Margins**: Configurable clearance around obstacles
3. **Workspace Centering**: Dynamic grid positioning for optimal coverage
4. **Time Limits**: Prevents infinite loops in complex environments

## Examples

Run the included examples:

```bash
python example_usage.py
```

This demonstrates:
- Basic pathfinding with obstacles
- GPS coordinate integration
- Performance testing with different configurations
- 3D visualization of planned paths

## Testing

```bash
pytest tests/
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add unit tests for new features
3. Update documentation for API changes
4. Test with both simulated and real UAV data

## License

MIT License - see LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{uav_astar_navigation,
  title={UAV A* Navigation System},
  author={UAV Navigation Team},
  year={2024},
  url={https://github.com/your-repo/uav-astar-navigation}
}
```
