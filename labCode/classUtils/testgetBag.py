import rosbag

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

bag_file = "../bags/square2-30-05-2025.bag"

## Open and read bag
bag = rosbag.Bag(bag_file)

pose_times = []
pose_vectors = []

obs_times = []
obs_data = []

# Read /pose messages
for topic, msg, t in bag.read_messages(topics=['/pose']):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orientation_z = msg.pose.pose.orientation.z
    linear_x = msg.twist.twist.linear.x
    angular_z = - msg.twist.twist.angular.z
    pose_vectors.append([t.to_sec(), x, y, orientation_z, linear_x, angular_z])
    pose_times.append(t.to_sec())

# Read /fiducial_transforms messages
for topic, msg, t in bag.read_messages(topics=['/fiducial_transforms']):
    observations = []
    for transform in msg.transforms:
        fiducial_id = transform.fiducial_id
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        observations.append([fiducial_id, tx, ty, tz])
    obs_data.append([t.to_sec(), observations])
    obs_times.append(t.to_sec())

bag.close()

## Align time to start at zero -------------------------------------------------------------------

time_bias = min(pose_times[0], obs_times[0])  
# print(f"Time bias: {time_bias}")

time_bias = min(pose_times[0], obs_times[0])
pose_times = [t - time_bias for t in pose_times]
obs_times = [t - time_bias for t in obs_times]

for entry in obs_data:
    entry[0] -= time_bias

# Also shift pose_vectors timestamps
for entry in pose_vectors:
    print(entry)
    entry[0] -= time_bias
    print(entry)

## Align time to start at zero -------------------------------------------------------------------

# Convert pose_vectors to numpy array for easier slicing
pose_vectors = np.array(pose_vectors)

# Interpolate pose values at obs_times
interp_x = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 1])
interp_y = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 2])
interp_orientation_z = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 3])
interp_linear_x = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 4])
interp_angular_z = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 5])


# Shift initial position to (0,0)

# Now you can do the same plots using matplotlib
import matplotlib.pyplot as plt

# Plot odometry path (X vs Y) with custom axis limits
plt.figure()
plt.plot(interp_x, interp_y, linewidth=1.5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.xlim([-1, 1])   # Change these values as needed
plt.ylim([-1, 1])   # Change these values as needed
plt.title('2D Odometry Path (/pose) - Origin Shifted')
plt.grid(True)
plt.axis('equal')
plt.show()


# Plot speeds
plt.figure()
plt.plot(obs_times, interp_linear_x, 'b-', label='Linear Speed')
plt.xlabel('Time (s)')
plt.ylabel('Linear Speed (m/s)', color='b')
plt.tick_params(axis='y', labelcolor='b')
plt.title('Linear Speed over Time')


plt.figure()
plt.plot(obs_times, interp_angular_z, 'r-', label='Angular Speed')
plt.xlabel('Time (s)')
plt.ylabel('Angular Speed (rad/s)', color='r')
plt.tick_params(axis='y', labelcolor='r')
plt.title('Angular Speeds over Time')
plt.show()
