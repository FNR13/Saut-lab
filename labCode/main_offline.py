import math
import numpy as np
import matplotlib.pyplot as plt



from scipy.linalg import orthogonal_procrustes


from classUtils.utils import wrap_angle_rad, update_paths, resample_paths, read_bag_data, draw_ellipse

from classAlgorithm.fastslam import FastSLAM

def main():
    bag_file = '/Users/afons/OneDrive/Ambiente de Trabalho/Faculdade/Mestrado/1º Ano/2º Sem/SAut/Saut-lab_1/Bags/square2-30-05-2025.bag'
    time, x, y, theta, velocity_vector, omega_vector, obs_data = read_bag_data(bag_file)

    # print("Observations data: ", omega_vector)
    
    # FastSLAM initialization
    robot_initial_pose = (0, 0, 0)

    N_PARTICLES = 100
    particles_odometry_uncertainty = (0.001, 0.01)
    landmarks_initial_uncertainty = 1
    Q_cov = np.diag([0.01, 0.05])  # Measurement noise covariance for range and bearing

    fastslam = FastSLAM(
        robot_initial_pose,
        N_PARTICLES,
        particles_odometry_uncertainty,
        landmarks_initial_uncertainty,
        Q_cov,
    )

    paths = np.zeros((N_PARTICLES, 1, 3), dtype=float)

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
        
        # Add the new pose to the path
        paths = update_paths(paths, new_pose)
        
        # Prepare observations for this timestep
        z_all = []
        for obs in obs_data[i][1]:
            marker_id, dx, dy, dz = obs
            # Convert to range and bearing relative to the robot pose
            # dx lateral distance(left/right is negative/positive), dz foward distance 
            rng = math.hypot(dz, dx)
            #bearing = wrap_angle_rad(math.atan2(dx, dz)) #Before
            bearing = wrap_angle_rad(math.atan2(dx, dz)+theta[i]-np.pi/2) #Now
            z_all.append([marker_id, rng, bearing])

        if z_all:
            fastslam.observation_update(z_all)
            fastslam.resampling()
            paths = resample_paths(paths, fastslam.resampled_indexes)

    # Select best particle and extract information
    selected_particle = fastslam.get_best_particle()
    best_path = fastslam.particles.index(selected_particle)
    landmarks_uncertainty = selected_particle.landmarks_position_covariance 
    B = selected_particle.landmarks_position
    B = np.array([b.flatten() for b in B])
    
    # Define landmarks and trajectories
    real_landmarks = np.array([
        (-0.08, -0.77), (0.24, 0.40), (-0.54, 1.33), (-0.52, 2.75),
        (-1.80, -0.77), (-1.30, 1.25), (-1.23, 2.80), (-1.30, 3.70),
        (-3.70, 3.35)
    ])
    square_Trajectory = np.array([
        (0, 0), (0, 4.5), (-1.8, 4.5), (-1.8, 0), (0, 0)
    ])
    L_Trajectory = np.array([
        (0, 0), (0, 4.5), (-3.9, 4.5)
    ])
    
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
        ax.plot(paths[idx, :, 0], -paths[idx, :, 1], alpha=0.3, linewidth=1)
    
    # Plot best path
    ax.plot(paths[best_path, :, 0], -paths[best_path, :, 1], label='Most probable path', color='blue', linewidth=2)
    
    # Plot landmarks, trajectories, and odometry
    ax.scatter(real_landmarks[:, 1], -real_landmarks[:, 0], c='green', marker='o', label='Real landmarks')
    ax.plot(square_Trajectory[:, 1], -square_Trajectory[:, 0], 'b--', label='Square trajectory')
    # ax.plot(L_Trajectory[:, 1], -L_Trajectory[:, 0], 'b--', label='L trajectory')
    ax.plot(x, y, 'r--', label='Odometry')
    
    # Plot uncertainty ellipses for landmarks
    if isinstance(B, np.ndarray):
        for i in range(len(landmarks_uncertainty)):
            ellipse = draw_ellipse(ax, B[i, :], landmarks_uncertainty[i])
            label = 'Estimated landmarks' if i == 0 else None
            if label:
                ellipse.set_label(label)
    
    # Final plot settings
    ax.set_title('FastSLAM (Bag Data)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='best')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()