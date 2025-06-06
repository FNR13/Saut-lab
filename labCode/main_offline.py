import math
import numpy as np
import matplotlib.pyplot as plt

from classUtils.utils import *

from classAlgorithm.fastslam import FastSLAM
from classAlgorithmRangeOnly.fastslam import FastSLAM_RO
from classAlgorithmBearingOnly.fastslam import FastSLAM_BO


# -----------------------------------------------------------------------------------------------------------------
def main():
    bag_file = "/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/square2-30-05-2025.bag"
    time, x, y, theta, velocity_vector, omega_vector, obs_data = read_bag_data(bag_file)
    camera_offset = - math.pi/2

    # FastSLAM initialization
    robot_initial_pose = (0, 0, 0)

    use_range_only_fastslam = False
    use_bearing_only_fastslam = False

    N_PARTICLES = 150
    particles_odometry_uncertainty = (0.005, 0.05)  # (speed, angular rate)
    landmarks_initial_uncertainty = 1

    if use_range_only_fastslam:
        print("Using Range Only FastSLAM")
        # Range Only FastSLAM parameters (from camera characterization file)    
        sensor_fov = 49.56      # Field of view of the camera in degrees   
        Q_cov = 5.64628409e-07  # Variance of the sensor for range

        # Range Only FastSLAM initialization
        fastslam = FastSLAM_RO(
            robot_initial_pose,
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
        Q_cov = 1.47227856e-10                  # Variance of the sensor for bearing

        # Bearing Only FastSLAM initialization
        fastslam = FastSLAM_BO(
            robot_initial_pose,
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
        Q_cov = np.diag([5.64628409e-07, 1.47227856e-10]) # Covariance matrix for range and bearing
        #Q_cov = np.diag([5.6, math.radians(10)]) # Covariance matrix for range and bearing

        fastslam = FastSLAM(
            robot_initial_pose,
            N_PARTICLES,
            particles_odometry_uncertainty,
            landmarks_initial_uncertainty,
            Q_cov,
        )

    ## Ground Truth landmarks and trajectories
    # Define landmarks and trajectories
    real_landmarks = np.array([
        (-0.08, -0.77), (0.24, 0.40), (-0.54, 1.33), (-0.52, 2.75),
        (-1.80, -0.77), (-1.30, 1.25), (-1.23, 2.80), (-1.30, 3.70),
        (-3.73, 3.35)
    ])
    real_landmarks_id = [11, 10, 9 ,8, 12, 13, 14, 15, 16]

    square_Trajectory = np.array([
        (0, -0.3), (0, 4.5), (-1.8, 4.5), (-1.8, -0.3), (0, -0.3)
    ])
    L_Trajectory = np.array([
        (0, 0), (0, 4.5), (-3.9, 4.5)
    ])

    # Array for saving all paths
    paths = np.zeros((N_PARTICLES, 1, 3), dtype=float)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Main loop: step through bag data
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
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
        z_all = []
        for obs in obs_data[i][1]:
            marker_id, dx, dy, dz = obs
            # Convert to range and bearing relative to the robot pose
            # dx lateral distance(left/right is negative/positive), dz foward distance 
            distance = math.hypot(dz, dx)
            #bearing = wrap_angle_rad(math.atan2(dx, dz)) #Before
            bearing = wrap_angle_rad(math.atan2(dx, dz) - camera_offset)

            if use_range_only_fastslam:
                # For Range Only FastSLAM, we only need the distance
                z_all.append([marker_id, distance])
            elif use_bearing_only_fastslam:
                # For Bearing Only FastSLAM, we only need the bearing
                z_all.append([marker_id, bearing])
            else:
                z_all.append([marker_id, distance, bearing])

        if z_all:
            fastslam.observation_update(z_all)
            fastslam.resampling()

            # Resample the paths in the same manner as the particles
            paths = resample_paths(paths, fastslam.resampled_indexes)

    # -----------------------------------------------------------------------------------------------------------------
    # Data prcessing 

        # Select best particle and extract information
    best_particle = fastslam.get_best_particle()
    best_path = fastslam.particles.index(best_particle)
    
    # ---------
    # Landmarks estimation
    identified_real_landmarks = []
    estimated_landmarks = []
    estimated_landmarks_covariance = []

    for id in best_particle.landmarks_id:
        idx_real = real_landmarks_id.index(id) if id in real_landmarks_id else None
        idx_estimated = best_particle.landmarks_id.index(id) if id in best_particle.landmarks_id else None
        
        estimated_landmarks.append(best_particle.landmarks_position[idx_estimated])
        estimated_landmarks_covariance.append(best_particle.landmarks_position_covariance[idx_estimated])

        identified_real_landmarks.append(real_landmarks[idx_real])
    
        # Convert to numpy array for vectorized indexing
    identified_real_landmarks = np.array(identified_real_landmarks)
    
    aligned_estimated_landmarks, estimated_center, real_center, Rotation = transform_landmarks(estimated_landmarks, identified_real_landmarks)
    
    estimated = np.array([
    [landmark[0][0], landmark[1][0]] for landmark in estimated_landmarks])
    
    aligned = np.array([
    [landmark[0], landmark[1]] for landmark in aligned_estimated_landmarks])

    RMSE = np.sqrt(np.mean(np.sum((aligned - identified_real_landmarks) ** 2, axis=1)))
    # ----------
    # Extract the best path

    most_probable_path = paths[best_path, :, :2]

        # Center the path using the same estimated_center as for landmarks
    centered_path = most_probable_path - estimated_center

        # Apply the same rotation and translation as for landmarks
        # NOTE: IF just one landmark was seen, the rotation will be identity, so it won work
    aligned_most_probable_path = (centered_path @ Rotation) + real_center

    # -----------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Data Visualization
    ground_truth_color = 'red'
    particle_color = 'black'
    aligned_estimation_color = 'blue'

    # --- Figure 1: Ground truth and aligned estimations ---
    fig, ax = plt.subplots()
    ax.plot(square_Trajectory[:, 0], square_Trajectory[:, 1], 
             'r--', label='Ground truth: Square trajectory', linewidth=2)
    # ax.plot(L_Trajectory[:, 0], L_Trajectory[:, 1], 
    #   'r--', label='Ground truth: L trajectory', linewidth=2)

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
        0.02, 0.98, f'RMSE: {RMSE:.3f}',
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
    
    if use_range_only_fastslam:
        ax1.set_title('Range Only FastSLAM (Odometry & Non-Aligned Estimation)')
    elif use_bearing_only_fastslam:
        ax1.set_title('Bearing Only FastSLAM (Odometry & Non-Aligned Estimation)')
    else:
        ax1.set_title('FastSLAM (Odometry & Non-Aligned Estimation)') 

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    
    plt.show()

# -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()