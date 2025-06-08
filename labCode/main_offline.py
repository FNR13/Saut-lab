import math
import numpy as np
import matplotlib.pyplot as plt
import time


from classUtils.utils import *

from classAlgorithm.fastslam import FastSLAM
from classAlgorithmRangeOnly.fastslam import FastSLAM_RO
from classAlgorithmBearingOnly.fastslam import FastSLAM_BO

# -----------------------------------------------------------------------------------------------------------------
def main():
    # Dataset 1 bags:
    # L-30-05-2025.bag
    # square2-30-05-2025.bag

    # Dataset 2 bags
    # 'straight.bag'
    # 'straight731.bag'
    # 'L.bag'
    # 'Lreturn.bag'
    # 'map1.bag'
    # 'map2.bag'

    bag_name = select_bag_file_from_list()
    if not bag_name:
        print("No bag file selected. Exiting.")
        return
    
    # bag_name  = 'Lreturn.bag'
    # Data lab conditions:
    dataset1, straight_trajectory, L_trajectory, square_trajectory, inverse, just_mapping, landmark297_change, camera_offset = set_data_conditions_from_bag(bag_name)
    # -----------------------------------------------------------------------------------------------------------------
    # FastSLAM initialization

        # Tipe of slam
    use_range_only_fastslam = False
    use_bearing_only_fastslam = False

        # Tuning parameters
    N_PARTICLES = 150
    particles_odometry_uncertainty = (0.005, 0.05)  # (speed, angular rate)
    landmarks_initial_uncertainty = 0.5
    Q_cov_range = 5.64628409e-07
    Q_cov_bearing = 1.47227856e-10
    Q_cov_range = 0.5
    Q_cov_bearing = 0.5


    if use_range_only_fastslam:
        print("Using Range Only FastSLAM")
        # Range Only FastSLAM parameters (from camera characterization file)    
        sensor_fov = 49.56   # Field of view of the camera in degrees   
        Q_cov = Q_cov_range  # Variance of the sensor for range

        # Range Only FastSLAM initialization
        fastslam = FastSLAM_RO(
            N_PARTICLES,
            particles_odometry_uncertainty,
            landmarks_initial_uncertainty,
            Q_cov, 
            sensor_fov,
        )

    elif use_bearing_only_fastslam:
        print("Using Bearing Only FastSLAM")
        # Bearing Only FastSLAM parameters (from camera characterization file)
        sensor_max_range = 5.35686663476375     # Maximum vision range of the camera 
        sensor_min_range = 0.33607217464420264  # Minimum vision range of the camera 
        Q_cov = Q_cov_bearing                  # Variance of the sensor for bearing

        # Bearing Only FastSLAM initialization
        fastslam = FastSLAM_BO(
            N_PARTICLES,
            particles_odometry_uncertainty,
            landmarks_initial_uncertainty,
            Q_cov, 
            sensor_max_range,
            sensor_min_range,
        )

    else:
        print("Using FastSLAM")
        # FastSLAM parameters (from camera characterization file)
        Q_cov = np.diag([Q_cov_range, Q_cov_bearing]) # Covariance matrix for range and bearing
        #Q_cov = np.diag([5.6, math.radians(10)]) # Covariance matrix for range and bearing

        fastslam = FastSLAM(
            N_PARTICLES,
            particles_odometry_uncertainty,
            landmarks_initial_uncertainty,
            Q_cov,
        )
    # -----------------------------------------------------------------------------------------------------------------
    # Ground Truth landmarks and trajectories
        # Define landmarks and trajectories for data set 1
    real_landmarks_map1 = np.array([
        (-0.08, -0.77), (0.24, 0.40), (-0.54, 1.33), (-0.52, 2.75),
        (-1.80, -0.77), (-1.30, 1.25), (-1.23, 2.80), (-1.30, 3.70),
        (-3.73, 3.35)
    ])
    real_landmarks_id_map1 = [11, 10, 9 ,8, 12, 13, 14, 15, 16]

    real_square_trajectory = np.array([
        (0, -0.3), (0, 4.5), (-1.8, 4.5), (-1.8, -0.3), (0, -0.3)
    ])

    real_L_trajectory_1 = np.array([
        (0, 0), (0, 4.5), (-3.9, 4.5)
    ])

    # Define landmarks and trajectories for data set 2
    # --- MAP ---
    real_landmarks_map2 = np.array([
        (0, -0.30),      # 297
        (-0.88, 0.26),   # 557
        (-0.70, 1.23),   # 934
        (0.80, 1.15),    # 582
        (0.80, 2.85),    # 206
        (0.80, 4.21),    # 545
        (0.80, 6.06),    # 360
        (0.50, 8.01),    # 433
        (-0.73, 8.01),   # 63
        (-0.90, 5.93),   # 337
        (-0.62, 5.43),   # 105
        (-0.41, 4.45),   # 952
        (-0.41, 2.33),   # 124
        (-2.00, 6.26),   # 836
        (-2.87, 7.99),   # 844
    ])
    real_landmarks_id_map2 = [297, 557, 934, 582, 206, 545, 360, 433, 63, 337, 105, 952, 124, 836, 844]

    # Straight Trajectory
    real_straight_trajectory = np.array([
        (0, 0), (0, 7.31)
    ])

    # L trajectory
    real_L_trajectory_2 = np.array([
        (0, 0), (0, 7.31), (-3.60, 7.31)
    ])

    # On just mapping this id was changed
    landmark_change = (-3.77,7.14)

    # Get right landmarks
    if dataset1:
        real_landmarks = real_landmarks_map1
        real_landmarks_id = real_landmarks_id_map1
    else:
        real_landmarks = real_landmarks_map2
        real_landmarks_id = real_landmarks_id_map2

    if straight_trajectory:
        real_trajectory = real_straight_trajectory
        legend = 'Straight Trajectory'

    elif L_trajectory:
        if dataset1:
            real_trajectory = real_L_trajectory_1
        else:
            real_trajectory = real_L_trajectory_2
        legend = 'L Trajectory'

    elif square_trajectory:
        real_trajectory = real_square_trajectory
        legend = 'Square Trajectory'
    elif just_mapping:
        pass
    else:
        print("Please select a trajectory option.")
        return

    if inverse:
        legend = " Inverse " + legend

    if landmark297_change: 
        real_landmarks_map2[0] = landmark_change
    # Data
    import os
    
    bag_file = "../Bags/" + bag_name
    if not os.path.exists(bag_file):
        bag_file = "../bags/" + bag_name
    if not os.path.exists(bag_file):
        raise FileNotFoundError("Bag: " + bag_name+ " not found in either 'Bags' or 'bags' directory.")
    
    time_, x, y, theta, velocity_vector, omega_vector, obs_data = read_bag_data(bag_file)

    # Array for saving all paths
    paths = np.zeros((N_PARTICLES, 1, 3), dtype=float)
    z_all = []

    # List for saving the iteration time
    iteration_time = []

    # -----------------------------------------------------------------------------------------------------------------------------
    # Main loop: step through bag data
    for i in range(1, len(time_)):
        start_clock = time.time() 
        dt = time_[i] - time_[i-1]
        velocity = velocity_vector[i]
        omega = omega_vector[i]

        # -----------------------------------------------------------------------------------------------------------------
        # ----------
        # FastSLAM step
        # ----------
        fastslam.predict_particles(velocity, omega, dt)

        new_pose = np.zeros((N_PARTICLES, 1, 3), dtype=float)
        for j, particle in enumerate(fastslam.particles):
            particle_pose = np.array([particle.x, particle.y, particle.theta])
            new_pose[j, 0, :] = particle_pose
        
        # Add the new pose to the path
        paths = update_paths(paths, new_pose)
        
        # Prepare observations for this timestep
        z = []
        for obs in obs_data[i][1]:
            marker_id, dx, dy, dz = obs
            # Convert to range and bearing relative to the robot pose
            # dx lateral distance(left/right is negative/positive), dz foward distance 
            distance = math.hypot(dz, dx)
            bearing = -wrap_angle_rad(math.atan2(dx, dz) - camera_offset)

            if use_range_only_fastslam:
                # For Range Only FastSLAM, we only need the distance
                z.append([marker_id, distance])
            elif use_bearing_only_fastslam:
                # For Bearing Only FastSLAM, we only need the bearing
                z.append([marker_id, bearing])
            else:
                z.append([marker_id, distance, bearing])

        if z:
            fastslam.observation_update(z)
            fastslam.resampling()

            # Resample the paths in the same manner as the particles
            paths = resample_paths(paths, fastslam.resampled_indexes)
            z_all.append(z)

        stop_clock = time.time()
        iteration_time.append(stop_clock - start_clock)

    # -----------------------------------------------------------------------------------------------------------------
    # Data processing

    # Select best particle and extract information
    best_particle = fastslam.get_best_particle()
    best_path = fastslam.particles.index(best_particle)
    
    # ---------
    # Landmarks estimation
    identified_real_landmarks = []
    estimated_landmarks = []
    estimated_landmarks_covariance = []
    
    for id in best_particle.landmarks_id:
        if id in real_landmarks_id:
            idx_real = real_landmarks_id.index(id)
            idx_estimated = best_particle.landmarks_id.index(id)
    
            # Flatten estimated landmark from shape (2, 1) to (2,)
            estimated_pos = best_particle.landmarks_position[idx_estimated].reshape(2,)
            estimated_landmarks.append(estimated_pos)
            estimated_landmarks_covariance.append(best_particle.landmarks_position_covariance[idx_estimated])
    
            # Append real landmark (already in shape (2,))
            identified_real_landmarks.append(real_landmarks[idx_real])
        else:
            print(f"Landmark ID {id} not found in real_landmarks_id — skipping.")
    
    # Convert to NumPy arrays of shape (N, 2)
    identified_real_landmarks = np.array(identified_real_landmarks)  # shape (N, 2)
    estimated_landmarks = np.array(estimated_landmarks)              # shape (N, 2)
    
    aligned_estimated_landmarks, estimated_center, real_center, Rotation = transform_landmarks(estimated_landmarks, identified_real_landmarks)
    
    estimated = estimated_landmarks.copy()
    
    aligned = aligned_estimated_landmarks.copy()

    # Performance meaurement for identification
    RMSE_landmarks = np.sqrt(np.mean(np.sum((aligned - identified_real_landmarks) ** 2, axis=1)))
    # ----------
    # Extract the best path

    most_probable_path = paths[best_path, :, :2]

        # Center the path using the same estimated_center as for landmarks
    centered_path = most_probable_path - estimated_center

        # Apply the same rotation and translation as for landmarks
        # NOTE: IF just one landmark was seen, the rotation will be identity, so it won work
    aligned_most_probable_path = (centered_path @ Rotation) + real_center
    # ----------
    # Performance meaurement for path 

    odometry_path = np.column_stack((x.copy(), y.copy()))
    RMSE_path = np.sqrt(np.mean(np.sum((odometry_path - most_probable_path) ** 2, axis=1)))
    # ----------
    # Save the iteration time to the computer

    iteration_time = np.array(iteration_time)

    if use_range_only_fastslam:
        file_name  = 'iteration_time_ROFastSLAM_'
    elif use_bearing_only_fastslam:
        file_name  = 'iteration_time_BOFastSLAM_'
    else:
        file_name  = 'iteration_time_FastSLAM_'

    file_name += (str(N_PARTICLES) + '_particles.npy')
    np.save(file_name, iteration_time)

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Data Visualization
    
    ground_truth_color = 'red'
    particle_color = 'black'
    aligned_estimation_color = 'blue'

    # --- Figure 1: Ground truth and aligned estimations ---
    fig, ax = plt.subplots()

    if not just_mapping:
        ax.plot(real_trajectory[:, 0], real_trajectory[:, 1], 
                'r--', label='Ground truth: ' + legend, linewidth=2)

    ax.scatter(identified_real_landmarks[:, 0], identified_real_landmarks[:, 1],
                facecolors='none', edgecolors=ground_truth_color, marker='o', label='Real Landmarks', linewidths=2)
    
    ax.plot(aligned_most_probable_path[:,0], aligned_most_probable_path[:,1], 
                aligned_estimation_color, label='Aligned Most Probable Path', linewidth=1)
    
    ax.scatter(aligned[:, 0], aligned[:, 1], 
                marker='x', c=aligned_estimation_color, label='Aligned Landmarks')

    if use_range_only_fastslam:
        ax.set_title('Range Only FastSLAM (Aligned Estimation)')
    elif use_bearing_only_fastslam:
        ax.set_title('Bearing Only FastSLAM (Aligned Estimation)')
    else:
        ax.set_title('FastSLAM (Aligned Estimation)') 

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    # Add RMSE text box to Figure 1
    ax.text(
        0.02, 0.98, f'RMSE landmarks: {RMSE_landmarks:.3f}',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Draw ellipses at aligned positions (uncertainty visualization)
    estimated_aligned_xy = aligned  # Already aligned XY positions
    uncertainty_list = []

    # Rotate covariance matrices
    for cov in estimated_landmarks_covariance:
        rotated_cov = Rotation @ cov @ Rotation.T
        uncertainty_list.append(rotated_cov)

    for i, aligned_pos in enumerate(estimated_aligned_xy):
        ellipse = draw_ellipse(ax, aligned_pos, uncertainty_list[i])
        if i == 0:
            ellipse.set_label("Uncertainty (aligned)")


    # --- Figure 2: Odometry and non-aligned estimations ---
    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, 'g--', label='Odometry')
    ax1.plot(most_probable_path[:,0], most_probable_path[:,1], 
                particle_color, label='Estimated Most Probable Path', linewidth=1)
    ax1.scatter(estimated[:, 0], estimated[:, 1], marker='x', c=particle_color, label='Estimated Landmarks')
    
    for i, (x, y) in enumerate(estimated):
        if i < len(best_particle.landmarks_id):
            ax1.annotate(str(best_particle.landmarks_id[i]), (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, color='black')

    if use_range_only_fastslam:
        ax1.set_title('Range Only FastSLAM (Odometry & Non-Aligned Estimation)')
    elif use_bearing_only_fastslam:
        ax1.set_title('Bearing Only FastSLAM (Odometry & Non-Aligned Estimation)')
    else:
        ax1.set_title('FastSLAM (Odometry & Non-Aligned Estimation)') 

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(loc='best', bbox_to_anchor=(0.5, 0.5))
    ax1.grid(True)

    # Add RMSE text box to Figure 2
    ax1.text(
        0.02, 0.98, f'RMSE path: {RMSE_path:.3f}',
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.show()
    

# -----------------------------------------------------------------------------------------------------------------
import os as oss
if __name__ == "__main__":
    oss.system('clear')
    main()