import math
import numpy as np
import matplotlib.pyplot as plt



from scipy.linalg import orthogonal_procrustes


from utils import wrap_angle_rad, update_paths, resample_paths, read_bag_data, draw_ellipse

from fastslam import FastSLAM

def main():
    bag_file = '/Users/usuario/Desktop/Máster/Autonomous systems/Project/Saut-lab/Bags/square2-30-05-2025.bag'
    time, x, y, theta, velocity_vector, omega_vector, obs_data = read_bag_data(bag_file)

    # FastSLAM initialization
    robot_initial_pose = (0, 0, 0)

    N_PARTICLES = 100
    particles_odometry_uncertainty = (0.001, 0.01)
    landmarks_initial_uncertainty = 1
    Q_cov = 0.01
    sensor_fov = 60 #vision range of the camera in º

    fastslam = FastSLAM(
        robot_initial_pose,
        N_PARTICLES,
        particles_odometry_uncertainty,
        landmarks_initial_uncertainty,
        Q_cov,
        sensor_fov,

    )

    paths = np.zeros((N_PARTICLES, 1, 3), dtype=float)
    best_path = 0
    landmarks_uncertainty = 0
    B = 0

    # Main loop: step through bag data
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        velocity = velocity_vector[i]
        omega = omega_vector[i]

        # FastSLAM motion update
        fastslam.predict_particles(velocity, omega, dt)

        new_pose = np.zeros((N_PARTICLES, 1, 3), dtype=float)
        for j, particle in enumerate(fastslam.particles):
            particle_pose = np.array([particle.x, particle.y, particle.theta])
            new_pose[j, 0, :] = particle_pose
        paths = update_paths(paths, new_pose)

        # Prepare observations for this timestep
        z_all = []
        for obs in obs_data[i][1]:
            marker_id, dx, dy, dz = obs
            # Convert to range relative to the robot pose

            rng = math.hypot(dx, dy)
            z_all.append([marker_id,rng])

        if z_all:
            fastslam.observation_update(z_all)
            fastslam.resampling()
            paths = resample_paths(paths, fastslam.resampled_indexes)

    # Select best particle and extract information
    selected_particle = fastslam.get_best_particle()
    best_path = fastslam.particles.index(selected_particle)
    landmarks_uncertainty = selected_particle.landmarks_position_covariance 
    estimated_landmarks = selected_particle.landmarks_position
    estimated_landmarks = np.array([b.flatten() for b in estimated_landmarks])
    
    ## Ground Truth landmarks and trajectories
    # Define landmarks and trajectories
    real_landmarks = np.array([
        (-0.08, -0.77), (0.24, 0.40), (-0.54, 1.33), (-0.52, 2.75),
        (-1.80, -0.77), (-1.30, 1.25), (-1.23, 2.80), (-1.30, 3.70),
    ])
    square_Trajectory = np.array([
        (0, -0.3), (0, 4.5), (-1.8, 4.5), (-1.8, -0.3), (0, -0.3)
    ])
    L_Trajectory = np.array([
        (0, 0), (0, 4.5), (-3.9, 4.5)
    ])
    
    ## Plot data
    # Create a single figure and axis
    fig, ax = plt.subplots()
    
    # Plot particle clouds at key frames
    for i in [0, len(time)//4, len(time)//2, 3*len(time)//4, -1]:
        x_part = [p.x for p in fastslam.particles]
        y_part = [p.y for p in fastslam.particles]
        ax.scatter(x_part, y_part, s=5, alpha=0.1, color='gray')
    
    # Plot multiple particle trajectories
    N_SAMPLE_PATHS = 10
    sampled_indices = np.random.choice(fastslam.num_particles, N_SAMPLE_PATHS, replace=False)
    for idx in sampled_indices:
        ax.plot(paths[idx, :, 0], paths[idx, :, 1], alpha=0.3, linewidth=1)
    
    ax.plot(square_Trajectory[:, 1], -square_Trajectory[:, 0], 'b--', label='Square trajectory')
    # ax.plot(L_Trajectory[:, 1], -L_Trajectory[:, 0], 'b--', label='L trajectory')
    ax.plot(x, y, 'r--', label='Odometry')
    
    # Plot landmarks, trajectories, and odometry
    ax.scatter(real_landmarks[:, 1], -real_landmarks[:, 0], c='green', marker='o', label='Real landmarks')

    # Convert real_landmarks from (-y, x) to (x, y)
    real_landmarks_xy = np.array([[b[1], -b[0]] for b in real_landmarks])

    #fig2, ax1 = plt.subplots()

    # Plot estimated landmarks (aligned or not)
    estimated_landmarks = np.array([[b[0], -b[1]] for b in estimated_landmarks])
    estimated_landmarks_alligned = estimated_landmarks  # or use align_by_centroid_and_pca if desired
    ax.scatter(estimated_landmarks_alligned[:, 0], -estimated_landmarks_alligned[:, 1], c='orange', marker='x', label='Estimated landmarks')

    # Plot uncertainty ellipses for landmarks
    '''
    if isinstance(estimated_landmarks, np.ndarray):
        for i in range(len(landmarks_uncertainty)):
            ellipse = draw_ellipse(ax, estimated_landmarks[i, :], landmarks_uncertainty[i])
            label = 'Estimated landmarks' if i == 0 else None
            if label:
                ellipse.set_label(label)'''

    # Plot best path
    ax.plot(paths[best_path, :, 0], paths[best_path, :, 1], label='Most probable path', color='blue', linewidth=2)

    # Final plot settings
    ax.set_title('FastSLAM (Bag Data)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    #ax1.set_title('Landmarks and Best Path')
    #ax1.set_xlabel('X')
    #ax1.set_ylabel('Y')
    #ax1.legend(loc='lower right')
    #plt.grid(True)
    #plt.show()

if __name__ == "__main__":
    main()