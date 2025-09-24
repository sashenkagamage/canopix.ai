#!/usr/bin/env python3
"""
Example usage of the UAV A* navigation system
Demonstrates GPS integration, obstacle avoidance, and path planning
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

from uav_navigation import (
    AStarUAV, 
    SearchConfig, 
    GPSCoordinateConverter,
    MockOccupancyMap
)


def visualize_path_3d(path, obstacles, start_pos, goal_pos):
    """Visualize the planned path in 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot path
    if len(path) > 0:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                'b-', linewidth=2, label='Planned Path')
        ax.scatter(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                  c='blue', s=20, alpha=0.6)
    
    # Plot start and goal
    ax.scatter(*start_pos, c='green', s=100, marker='o', label='Start')
    ax.scatter(*goal_pos, c='red', s=100, marker='*', label='Goal')
    
    # Plot obstacles
    for center, size in obstacles:
        # Draw a simple box representation
        x_range = [center[0] - size[0]/2, center[0] + size[0]/2]
        y_range = [center[1] - size[1]/2, center[1] + size[1]/2]
        z_range = [center[2] - size[2]/2, center[2] + size[2]/2]
        
        # Draw edges of the box
        for x in x_range:
            for y in y_range:
                ax.plot([x, x], [y, y], z_range, 'r-', alpha=0.3)
        for x in x_range:
            for z in z_range:
                ax.plot([x, x], y_range, [z, z], 'r-', alpha=0.3)
        for y in y_range:
            for z in z_range:
                ax.plot(x_range, [y, y], [z, z], 'r-', alpha=0.3)
    
    ax.set_xlabel('East (m)')
    ax.set_ylabel('North (m)')
    ax.set_zlabel('Up (m)')
    ax.set_title('UAV A* Path Planning')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = 50
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range])
    
    plt.tight_layout()
    plt.show()


def example_simple_navigation():
    """Basic example with mock obstacles"""
    print("=== Simple UAV Navigation Example ===")
    
    # Create configuration
    config = SearchConfig(
        step_size=1.0,          # 1 meter grid resolution
        time_limit=5.0,         # 5 second search limit
        pool_size=(100, 100, 50) # 100x100x50 meter search space
    )
    
    # Initialize A* planner
    planner = AStarUAV(config)
    
    # Create mock occupancy map with some obstacles
    occupancy_map = MockOccupancyMap()
    
    # Add some obstacles (center, size)
    obstacles = [
        (np.array([10, 10, 15]), np.array([5, 5, 10])),  # Building
        (np.array([25, 15, 8]), np.array([3, 8, 5])),    # Tower
        (np.array([35, 30, 12]), np.array([6, 4, 8])),   # Structure
        (np.array([15, 35, 6]), np.array([4, 6, 3]))     # Low obstacle
    ]
    
    for center, size in obstacles:
        occupancy_map.add_obstacle(center, size)
    
    planner.set_occupancy_map(occupancy_map)
    
    # Define start and goal positions
    start_pos = np.array([0, 0, 10])    # Start at origin, 10m altitude
    goal_pos = np.array([40, 40, 20])   # Fly to (40,40) at 20m altitude
    
    print(f"Planning path from {start_pos} to {goal_pos}")
    print(f"Grid resolution: {config.step_size}m")
    print(f"Search space: {config.pool_size}")
    
    # Plan the path
    start_time = time.time()
    success, path, stats = planner.search(start_pos, goal_pos)
    planning_time = time.time() - start_time
    
    # Display results
    if success:
        print(f"✅ Path found!")
        print(f"Path length: {len(path)} waypoints")
        print(f"Total distance: {sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1)):.2f}m")
        print(f"Planning time: {planning_time:.3f}s")
        print(f"Search stats: {stats}")
        
        # Print first few waypoints
        print("\nFirst 5 waypoints:")
        for i, waypoint in enumerate(path[:5]):
            print(f"  {i+1}: [{waypoint[0]:.1f}, {waypoint[1]:.1f}, {waypoint[2]:.1f}]")
        
        if len(path) > 5:
            print(f"  ... and {len(path)-5} more waypoints")
            print(f"  Final: [{path[-1][0]:.1f}, {path[-1][1]:.1f}, {path[-1][2]:.1f}]")
        
        # Visualize if matplotlib is available
        try:
            visualize_path_3d(path, obstacles, start_pos, goal_pos)
        except ImportError:
            print("Matplotlib not available for visualization")
            
    else:
        print("❌ No path found!")
        print(f"Planning time: {planning_time:.3f}s")
        print(f"Error: {stats.get('error', 'Unknown error')}")


def example_gps_integration():
    """Example with GPS coordinates"""
    print("\n=== GPS Integration Example ===")
    
    # Define GPS origin (example: somewhere in California)
    origin_lat, origin_lon, origin_alt = 37.7749, -122.4194, 100.0
    
    # Initialize GPS converter
    gps_converter = GPSCoordinateConverter(origin_lat, origin_lon, origin_alt)
    
    # Define GPS waypoints
    start_gps = (37.7750, -122.4194, 120.0)  # Slightly north, 20m higher
    goal_gps = (37.7748, -122.4190, 150.0)   # Slightly south and east, 50m higher
    
    print(f"Origin GPS: ({origin_lat:.6f}, {origin_lon:.6f}, {origin_alt:.1f})")
    print(f"Start GPS: ({start_gps[0]:.6f}, {start_gps[1]:.6f}, {start_gps[2]:.1f})")
    print(f"Goal GPS: ({goal_gps[0]:.6f}, {goal_gps[1]:.6f}, {goal_gps[2]:.1f})")
    
    # Convert to local coordinates
    start_local = gps_converter.gps_to_local(*start_gps)
    goal_local = gps_converter.gps_to_local(*goal_gps)
    
    print(f"Start local: [{start_local[0]:.2f}, {start_local[1]:.2f}, {start_local[2]:.2f}]")
    print(f"Goal local: [{goal_local[0]:.2f}, {goal_local[1]:.2f}, {goal_local[2]:.2f}]")
    
    # Plan path in local coordinates
    config = SearchConfig(step_size=2.0, pool_size=(50, 50, 30))
    planner = AStarUAV(config)
    
    # Simple occupancy map for this example
    occupancy_map = MockOccupancyMap()
    occupancy_map.add_obstacle(np.array([20, 5, 25]), np.array([8, 8, 15]))
    planner.set_occupancy_map(occupancy_map)
    
    success, path_local, stats = planner.search(start_local, goal_local)
    
    if success:
        print(f"✅ Path found in local coordinates!")
        
        # Convert path back to GPS
        print("\nPath in GPS coordinates:")
        for i, local_pos in enumerate(path_local[::5]):  # Every 5th waypoint
            gps_pos = gps_converter.local_to_gps(local_pos)
            print(f"  Waypoint {i*5+1}: ({gps_pos[0]:.6f}, {gps_pos[1]:.6f}, {gps_pos[2]:.1f})")
    else:
        print("❌ No path found!")


def example_performance_test():
    """Performance test with different configurations"""
    print("\n=== Performance Test ===")
    
    configurations = [
        ("Coarse", SearchConfig(step_size=2.0, pool_size=(50, 50, 25))),
        ("Medium", SearchConfig(step_size=1.0, pool_size=(80, 80, 40))),
        ("Fine", SearchConfig(step_size=0.5, pool_size=(100, 100, 50))),
    ]
    
    # Create a challenging environment
    occupancy_map = MockOccupancyMap()
    obstacles = [
        (np.array([15, 15, 10]), np.array([8, 8, 8])),
        (np.array([30, 20, 15]), np.array([5, 10, 10])),
        (np.array([20, 35, 8]), np.array([6, 5, 6])),
        (np.array([35, 35, 12]), np.array([4, 8, 8])),
    ]
    for center, size in obstacles:
        occupancy_map.add_obstacle(center, size)
    
    start_pos = np.array([5, 5, 5])
    goal_pos = np.array([45, 45, 20])
    
    print(f"Test scenario: {start_pos} → {goal_pos}")
    print("Configuration | Success | Time(ms) | Path Length | Nodes Explored")
    print("-" * 70)
    
    for name, config in configurations:
        planner = AStarUAV(config)
        planner.set_occupancy_map(occupancy_map)
        
        start_time = time.time()
        success, path, stats = planner.search(start_pos, goal_pos)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        path_length = len(path) if success else 0
        nodes_explored = stats.get('nodes_explored', 0)
        
        status = "✅" if success else "❌"
        print(f"{name:12} | {status:7} | {elapsed_time:8.1f} | {path_length:11} | {nodes_explored:14}")


if __name__ == "__main__":
    # Run all examples
    example_simple_navigation()
    example_gps_integration()
    example_performance_test()
    
    print("\n=== Examples Complete ===")
    print("Next steps:")
    print("1. Integrate with your LiDAR/SLAM system")
    print("2. Connect to UAV flight controller")
    print("3. Add real-time replanning capabilities")
    print("4. Implement path smoothing for better flight dynamics")
