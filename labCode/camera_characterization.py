import numpy as np
import math
from classUtils.utils import *

# Script for camera characterization 
#--------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------

# Covariance of the camera
camera_offset = 0


paths = []
paths.append('/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/static0_5meters.bag')
paths.append('/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/static0_83meters.bag')
paths.append('/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/static0_8351meters.bag')
paths.append('/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/static1meter.bag')
paths.append('/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/static2meters.bag')

root = 'Q_cov'
Q_covariances = {}

# Read bag file from camera characterization test 

for i in range(len(paths)):

    name = f"{root}_{i}" 
    
    bag_file = paths[i]
    time, obs_data = read_bag_data(bag_file)

    # Obtain measurements 
    measurements_range = []
    measurements_bear = []

    for t in range(1,len(time)):
        for obs in obs_data[t][1]:
            marker_id, dx, dy, dz = obs
            # We make sure that only the landmark intended for the test is the one used for the analysis 
            if marker_id == 297:
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
    Q_covariances[name] =  np.cov(measurements) #.cov receives the variables in rows and the different measurements in columns    
    print(Q_covariances[name],'\n')


count = 0
for key, cov_matrix in Q_covariances.items():
    sum_cov = np.zeros_like(cov_matrix)  # matriz de ceros con misma forma
    sum_cov += cov_matrix
    count += 1

average_cov = sum_cov / count
Q_cov = average_cov

print("Q_cov from statistics analysis:\n", Q_cov)


#--------------------------------------------------------------------------------------------------------
# Max range of the camera
bag_file = '/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/range_max.bag'
time, obs_data = read_bag_data(bag_file)

# Obtain measurements 
measurements_range = []
measurements_bear = []

max_range = 0
min_range = 0
seen = False
for t in range(1,len(time)):
    for obs in obs_data[t][1]:
        marker_id, dx, dy, dz = obs
        # We make sure that only the landmark intended for the test is the one used for the analysis 
        if marker_id == 297:
            max_range = math.hypot(dz, dx)

            if not seen:
                min_range = max_range
                seen = True

print('\nMin range of the camera:',min_range)
print('Max range of the camera:',max_range)


#--------------------------------------------------------------------------------------------------------
# Field of view of the camera
bag_file = '/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/stbearing_max.bag'
time, obs_data = read_bag_data(bag_file)

# Obtain measurements 
measurements_range = []
measurements_bear = []

max_bearing = 0
for t in range(1,len(time)):
    for obs in obs_data[t][1]:
        marker_id, dx, dy, dz = obs
        # We make sure that only the landmark intended for the test is the one used for the analysis 
        if marker_id == 297:
            max_bearing = wrap_angle_rad(math.atan2(dx, dz) - camera_offset)

fov = 2*abs(max_bearing)
print(f"Field of view of the camera: {fov:.5f} rad ({math.degrees(fov):.2f}º)\n")

            



















