import math
import numpy as np
import matplotlib.pyplot as plt



from scipy.linalg import orthogonal_procrustes


from utils import wrap_angle_rad, update_paths, resample_paths, read_bag_data

from fastslam import FastSLAM

def main():
    bag_file = '/home/ricardo/saut/OfflineEstimation/lab2testWithId.bag'
    time, x, y, theta, velocity_vector, omega_vector, obs_data = read_bag_data(bag_file)

    # FastSLAM initialization
    robot_initial_pose = (0, 0, 0)

    N_PARTICLES = 100
    particles_odometry_uncertanty = (0.001, 0.01)
    landmarks_initial_uncertanty = 1
    Q_cov = np.diag([0.01, 0.05])  # Measurement noise covariance for range and bearing

    fastslam = FastSLAM(
        robot_initial_pose,
        N_PARTICLES,
        particles_odometry_uncertanty,
        landmarks_initial_uncertanty,
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
        paths = update_paths(paths, new_pose)

        # Prepare observations for this timestep
        z_all = []
        for obs in obs_data[i][1]:
            marker_id, dx, dy, dz = obs
            # Convert to range and bearing relative to the robot pose

            rng = math.hypot(dx, dy)
            bearing = wrap_angle_rad(math.atan2(dy, dx))
            z_all.append([marker_id, rng, bearing])

        if z_all:
            fastslam.observation_update(z_all)
            fastslam.resampling()
            paths = resample_paths(paths, fastslam.resampled_indexes)

    # Plotting results
    selected_particle = fastslam.get_best_particle()
    best_path = fastslam.particles.index(selected_particle)
    B = selected_particle.landmarks_position
    B = np.array([b.flatten() for b in B])
    x_mapped = B[:, 0]
    y_mapped = -B[:, 1]

    plt.plot(paths[best_path, :, 0], -paths[best_path, :, 1], label='Most probable path')
    plt.scatter(x_mapped, y_mapped, c='red', marker='x', label='Landmarks estimation')
    plt.title('FastSLAM (Bag Data)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()