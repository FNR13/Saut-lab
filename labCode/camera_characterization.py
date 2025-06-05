import numpy as np
import math
from classUtils.utils import *

# Script for camera characterization 
#---------------------------------------------------------------------

# Read bag file from camera characterization test
bag_file = "/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/square2-30-05-2025.bag"
time, x, y, theta, velocity_vector, omega_vector, obs_data = read_bag_data(bag_file)

camera_offset = 0

# Obtain measurements 
measurements_range = []
measurements_bear = []

for t in range(1,len(time)):
    for obs in obs_data[t][1]:
        marker_id, dx, dy, dz = obs
        distance = math.hypot(dz, dx)
        bearing = wrap_angle_rad(math.atan2(dx, dz) - camera_offset)
        measurements_range.append(distance)
        measurements_bear.append(bearing)

measurements_range = np.array(measurements_range)
measurements_bear = np.array(measurements_bear)
measurements = np.vstack((measurements_range, measurements_bear))  # shape: (2, N)


# Compute covarinace matrix
#1st column: range
#2nd column: bearing 
Q_cov =  np.cov(measurements).copy() #.cov receives the variables in rows and the different measurements in columns    