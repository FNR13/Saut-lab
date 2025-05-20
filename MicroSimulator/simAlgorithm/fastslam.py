import math
import numpy as np

from simAlgorithm.particle import Particle

from utils import wrap_angle_rad


class FastSLAM:
    def __init__(self, robot_initial_pose, num_particles, particles_odometry_uncertanty, landmarks_initial_uncertenty, Q_cov):

        self.robot_initial_pose = robot_initial_pose  # (x, y, theta)

        # Particle Filter parameters
        self.num_particles = num_particles

        self.particles_odometry_uncertanty = particles_odometry_uncertanty
        self.landmarks_initial_uncertenty = landmarks_initial_uncertenty

        self.Q_cov = Q_cov  # Measurement noise covariance for range and bearing

        self.particles = self._init_particles()

    # Particle Filter Management
    def _init_particles(self):
        return [
            Particle(self.particles_odometry_uncertanty, self.landmarks_initial_uncertenty, self.Q_cov)
            for _ in range(self.num_particles)
        ]

    def predict_particles(self, v, omega, dt):
        for particle in self.particles:
            particle.motion_update(v, omega, dt)
    
    # Landmark Management
    def observation_update(self, z_all):

        for observation in z_all:

            marker_id = observation[1]
            z = np.array(observation[:2]).reshape(2, 1)

            for particle in self.particles:
                particle.landmark_update(marker_id, z)


    def resampling(self):
        self.normalize_weight()

        N = self.num_particles
        weights = np.array([p.weight for p in self.particles])

        r = np.random.uniform(0, 1.0 / N)  # Random offset
        c = weights[0]                     # Cumulative sum
        i = 0
        new_particles = []
        for m in range(N):
            U = r + m * (1.0 / N)
            while U > c:
                i += 1
                c += weights[i]
            p = self.particles[i]
            new_p = type(p)(
                p.odometry_noise, 
                p.landmark_uncertanty, 
                p.Q_cov
            )
            # Copy pose and weight
            new_p.x = p.x
            new_p.y = p.y
            new_p.theta = p.theta
            new_p.weight = 1.0  # Reset weight after resampling

            # Deep copy all landmark-related attributes
            new_p.landmarks_id = p.landmarks_id.copy()
            new_p.landmarks_position = [np.copy(pos) for pos in p.landmarks_position]
            new_p.landmarks_position_covariance = [np.copy(cov) for cov in p.landmarks_position_covariance]
            new_p.landmarks_observation_count = p.landmarks_observation_count.copy()
            new_p.landmarks_EKF = [ekf for ekf in p.landmarks_EKF]

            new_particles.append(new_p)
        self.particles = new_particles
    
    def calc_final_state(self):
        x_est = np.zeros((3, 1))
        self.normalize_weight()
        for p in self.particles:
            x_est[0, 0] += p.weight * p.x
            x_est[1, 0] += p.weight * p.y
            x_est[2, 0] += p.weight * p.theta
        x_est[2, 0] = wrap_angle_rad(x_est[2, 0])
        return x_est

    def get_best_particle(self):
        return max(self.particles, key=lambda p: p.weight)
    

    def normalize_weight(self):
        total_weight = sum(p.weight for p in self.particles)
        if total_weight == 0:
            total_weight = 1e-8
        for p in self.particles:
            p.weight /= total_weight