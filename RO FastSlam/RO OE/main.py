import math
import numpy as np
import matplotlib.pyplot as plt



from scipy.linalg import orthogonal_procrustes


from utils import wrap_angle_rad, update_paths, resample_paths, read_bag_data, draw_ellipse

from fastslam import FastSLAM

def main():
    bag_file = '/home/ricardo/saut/OfflineEstimation/lab2testWithId.bag'
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

    # Plotting results
    selected_particle = fastslam.get_best_particle()
    best_path = fastslam.particles.index(selected_particle)
    landmarks_uncertainty = selected_particle.landmarks_position_covariance 
    B = selected_particle.landmarks_position
    B = np.array([b.flatten() for b in B])

    # Draw the most probable path and the estimated landmark positions 
    if isinstance(B, np.ndarray):

        fig, ax = plt.subplots()
        plt.plot(paths[best_path,:,0],-paths[best_path,:,1], label='Most probable path')
        for i in range(len(landmarks_uncertainty)):
            ellipse = draw_ellipse(ax, B[i,:], landmarks_uncertainty[i])
            label = 'Estimated landmarks' if i == 0 else None
            if label:
                ellipse.set_label(label)
        
    plt.title('FastSLAM (Bag Data)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc = 'best')
    plt.show()

if __name__ == "__main__":
    main()