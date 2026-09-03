import airsim
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Configure matplotlib for better display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 用于记录飞行路径
flight_path = []

def record_position():
    """Record current position"""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    # Convert Z to positive altitude for visualization (Z=-10 means 10m high)
    flight_path.append([pos.x_val, pos.y_val, -pos.z_val])
    return pos

def smooth_flight_to_waypoint(x, y, z, yaw_angle):
    """Smooth flight to waypoint by recording intermediate positions"""
    # Move to waypoint
    client.moveToPositionAsync(
        x, y, z, 
        velocity=2,
        yaw_mode=airsim.YawMode(False, math.degrees(yaw_angle))
    ).join()
    
    # Record positions during flight for smoother visualization
    for _ in range(3):  # Record a few intermediate points
        time.sleep(0.1)
        record_position()

print("Takeoff...")
# Record ground position before takeoff
state = client.getMultirotorState()
pos = state.kinematics_estimated.position
flight_path.append([pos.x_val, pos.y_val, -pos.z_val])  # Record starting point on ground

# Takeoff to 10m altitude
client.takeoffAsync().join()
time.sleep(0.3)
record_position()

# Move to stable altitude
client.moveToZAsync(-10, 3).join()  # Negative Z means up, move to 10m altitude
time.sleep(0.3)
record_position()  # Record starting position at stable altitude

print("Starting exploration flight...")
start_time = time.time()
exploration_duration = 60  # 60 seconds exploration (1 minute)

# Beautiful spiral exploration pattern with varying altitude
# Starting from near origin (0,0,-10) to ensure smooth trajectory
waypoints = [
    (2, 0, -10),      # Start close to origin, move slightly north
    (4, 2, -11),      # Northeast, climb slightly
    (6, 4, -12),      # Continue northeast, climb
    (7, 7, -13),      # East-northeast, higher
    (6, 10, -14),     # East, climb more
    (4, 11, -15),     # Southeast, higher
    (0, 12, -16),     # South, highest point
    (-4, 11, -15),    # Southwest, descend
    (-6, 9, -14),     # West, lower
    (-8, 6, -13),     # West-southwest, descend
    (-9, 2, -12),     # Southwest, lower
    (-10, -2, -11),   # West, lower
    (-9, -6, -10),    # Northwest, back to mid altitude
    (-7, -9, -9),     # North, descend more
    (-4, -11, -8),    # North-northeast, lower
    (0, -12, -7),     # East, lowest point
    (4, -11, -8),     # East-southeast, climb back
    (7, -9, -9),      # Southeast, higher
    (9, -6, -10),     # South, higher
    (10, -2, -11),    # South-southwest, climb
    (9, 2, -12),      # West, higher
    (7, 6, -13),      # West-northwest, higher
    (5, 9, -14),      # Northwest, climb
    (2, 10, -15),     # North, higher
    (0, 10, -15),     # Center line, maintain
    (-2, 8, -14),     # Spiral inward, descend
    (0, 6, -13),      # Continue inward
    (2, 4, -12),      # Spiral to center
    (1, 2, -11),      # Almost at center
    (0, 1, -10),      # Return near origin
]

# Calculate time allocation per waypoint
time_per_waypoint = exploration_duration / len(waypoints)

for i, waypoint in enumerate(waypoints):
    elapsed_time = time.time() - start_time
    if elapsed_time >= exploration_duration:
        break
    
    x, y, z = waypoint
    print(f"Flying to waypoint {i+1}/{len(waypoints)}: North{x}m, East{y}m, Alt{-z}m")
    
    # Calculate flight direction yaw angle
    current_pos = client.getMultirotorState().kinematics_estimated.position
    dx = x - current_pos.x_val
    dy = y - current_pos.y_val
    yaw = math.atan2(dy, dx)  # Calculate heading angle
    
    # Smooth flight to waypoint with continuous position recording
    smooth_flight_to_waypoint(x, y, z, yaw)
    
    # Record arrival at waypoint
    pos = record_position()
    print(f"Current position: x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")
    
    # Stop and rotate to explore surroundings (skip rotation for smoother trajectory)
    print(f"Arrived at waypoint {i+1}, brief pause...")
    time.sleep(0.5)  # Brief pause
    record_position()
    
    print(f"Waypoint {i+1} exploration complete\n")

print("\nExploration complete, flying to final target position...")
# Final target: South 15m (x=-15), West 10m (y=-10), Altitude 10m (z=-10)
final_x, final_y, final_z = -15, -10, -10

# Calculate final heading
current_pos = client.getMultirotorState().kinematics_estimated.position
dx = final_x - current_pos.x_val
dy = final_y - current_pos.y_val
final_yaw = math.atan2(dy, dx)

print(f"Flying to final position: South 15m, West 10m, Altitude 10m")
client.moveToPositionAsync(
    final_x, final_y, final_z, 
    velocity=2,
    yaw_mode=airsim.YawMode(False, math.degrees(final_yaw))
).join()

# Record some intermediate points for smooth visualization
for _ in range(3):
    time.sleep(0.2)
    record_position()

print("Arrived at final position, hovering...")
record_position()  # Record final position
time.sleep(3)  # Hover at final position for 3 seconds

# Get final state
final_state = client.getMultirotorState()
print(f"\nFinal position: x={final_state.kinematics_estimated.position.x_val:.2f}, "
      f"y={final_state.kinematics_estimated.position.y_val:.2f}, "
      f"z={final_state.kinematics_estimated.position.z_val:.2f}")

# Landing
print("\nMission complete, preparing to land...")
client.landAsync(timeout_sec=10).join()  # Faster landing with timeout
time.sleep(0.5)

# Record final ground position
state = client.getMultirotorState()
pos = state.kinematics_estimated.position
flight_path.append([pos.x_val, pos.y_val, -pos.z_val])

# Release control
client.armDisarm(False)
client.enableApiControl(False)
print("Script execution complete!")

# Generate 3D flight path visualization
print("\nGenerating 3D flight path visualization...")
flight_path = np.array(flight_path)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Draw flight path with smooth curve
ax.plot(flight_path[:, 0], flight_path[:, 1], flight_path[:, 2], 
        'b-', linewidth=3, label='Flight Path', alpha=0.7)

# Mark start and end points
ax.scatter(flight_path[0, 0], flight_path[0, 1], flight_path[0, 2], 
           c='green', s=300, marker='o', label='Start Point', 
           edgecolors='darkgreen', linewidths=3, zorder=5)
ax.scatter(flight_path[-1, 0], flight_path[-1, 1], flight_path[-1, 2], 
           c='red', s=400, marker='*', label='End Point', 
           edgecolors='darkred', linewidths=3, zorder=5)

# Mark key waypoints along the path
waypoint_indices = [i for i in range(0, len(flight_path), max(1, len(flight_path)//20))]
ax.scatter(flight_path[waypoint_indices, 0], 
           flight_path[waypoint_indices, 1], 
           flight_path[waypoint_indices, 2],
           c='orange', s=80, marker='o', alpha=0.6, 
           label='Waypoints', edgecolors='darkorange', linewidths=1.5)

# Set axis labels with larger fonts
ax.set_xlabel('X Axis (North-South) [m]', fontsize=14, labelpad=15, fontweight='bold')
ax.set_ylabel('Y Axis (East-West) [m]', fontsize=14, labelpad=15, fontweight='bold')
ax.set_zlabel('Z Axis (Altitude) [m]', fontsize=14, labelpad=15, fontweight='bold')

# Adjust the axis limits for better visualization without forcing equal aspect
x_range = flight_path[:, 0].max() - flight_path[:, 0].min()
y_range = flight_path[:, 1].max() - flight_path[:, 1].min()
z_range = flight_path[:, 2].max() - flight_path[:, 2].min()

# Add some padding
padding = 0.2
ax.set_xlim(flight_path[:, 0].min() - x_range * padding, 
            flight_path[:, 0].max() + x_range * padding)
ax.set_ylim(flight_path[:, 1].min() - y_range * padding, 
            flight_path[:, 1].max() + y_range * padding)
ax.set_zlim(flight_path[:, 2].min() - z_range * padding, 
            flight_path[:, 2].max() + z_range * padding)

# Set title
ax.set_title('Drone Flight Path 3D Trajectory', fontsize=20, fontweight='bold', pad=30)

# Add legend with better styling
ax.legend(loc='upper left', fontsize=12, framealpha=0.95, shadow=True)

# Add grid with better styling
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Set optimal viewing angle for 3D trajectory
ax.view_init(elev=20, azim=135)

# Display statistics
total_distance = np.sum(np.sqrt(np.sum(np.diff(flight_path, axis=0)**2, axis=1)))
print(f"\nFlight Statistics:")
print(f"- Total path points: {len(flight_path)}")
print(f"- Total flight distance: {total_distance:.2f} meters")
print(f"- Start coordinates: ({flight_path[0, 0]:.2f}, {flight_path[0, 1]:.2f}, {flight_path[0, 2]:.2f})")
print(f"- End coordinates: ({flight_path[-1, 0]:.2f}, {flight_path[-1, 1]:.2f}, {flight_path[-1, 2]:.2f})")
print(f"- Altitude range: {flight_path[:, 2].min():.2f}m to {flight_path[:, 2].max():.2f}m (ground=0m)")

plt.tight_layout()
plt.savefig('drone_flight_path.png', dpi=300, bbox_inches='tight')
print("\nFlight path visualization saved as: drone_flight_path.png")
plt.show()