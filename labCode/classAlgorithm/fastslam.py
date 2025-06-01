import math
import numpy as np
import copy

# Debugging imports
try:
    # Relative import for normal package usage
    from .particle import Particle
except ImportError:
    # Absolute import fallback for direct script testing
    from particle import Particle

try:
    # Relative import for normal package usage
    from classUtils.utils import wrap_angle_rad
    
except ModuleNotFoundError:
    # Absolute import fallback for direct script testing
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from classUtils.utils import wrap_angle_rad


class FastSLAM:
    def __init__(self, robot_initial_pose, num_particles, particles_odometry_uncertainty, landmarks_initial_uncertainty, Q_cov):

        self.robot_initial_pose = robot_initial_pose  # (x, y, theta)

        # Particle Filter parameters
        self.num_particles = num_particles
        self.particles_odometry_uncertainty = particles_odometry_uncertainty
        self.landmarks_initial_uncertainty = landmarks_initial_uncertainty

        self.Q_cov = Q_cov  # Measurement noise covariance for range and bearing

        self.particles = self._init_particles()
        self.resampled_indexes = [] #list containing the resampled indexes 
        self.best_index = 0


    # Particle Filter Management
    def _init_particles(self):
        return [
            Particle(self.particles_odometry_uncertainty, self.landmarks_initial_uncertainty, self.Q_cov)
            for _ in range(self.num_particles)
        ]

    def predict_particles(self, v, omega, dt):
        for particle in self.particles:
            particle.motion_update(v, omega, dt)
    
    # Landmark Management
    def observation_update(self, z_all):

        for observation in z_all:

            marker_id = observation[0]
            z = np.array(observation[1:]).reshape(2, 1)

            for particle in self.particles:
                particle.landmark_update(marker_id, z)


    def resampling(self):
        self.normalize_weight()

        N = self.num_particles
        weights = np.array([p.weight for p in self.particles]) #Normalized weights
        #print('weights',weights)
        r = np.random.uniform(0, 1.0 / N)  # Random offset
        c = weights[0]                     # Cumulative sum
        i = 0
        new_particles = []
        self.resampled_indexes = []

        for m in range(N):
            U = r + m * (1.0 / N)
            while U > c:
                i += 1
                c += weights[i]

            self.resampled_indexes.append(i) #List with the resampled indexes 
            p = self.particles[i]
            new_p = type(p)(
                p.odometry_noise, 
                p.landmark_uncertainty, 
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
            new_p.landmarks_EKF = [copy.deepcopy(ekf) for ekf in p.landmarks_EKF]

            new_particles.append(new_p)
        self.particles = new_particles
        #print('indexes',self.resampled_indexes)

    
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

def test_fastslam():
    import numpy as np

    # Initial robot pose
    robot_initial_pose = [0.0, 0.0, 0.0]
    num_particles = 3
    particles_odometry_uncertainty = (0.1, 0.01)  # (speed, angular rate)
    landmarks_initial_uncertainty = 10  # Initial uncertainty for landmarks
    Q_cov = np.diag([20.0, np.radians(30)])  # Measurement noise for fast slam - for range and bearing

    # Create FastSLAM object
    slam = FastSLAM(robot_initial_pose, num_particles, particles_odometry_uncertainty, landmarks_initial_uncertainty, Q_cov)

    # Simulate motion update
    v = 1
    omega = 0
    dt = 1.0

    robot_initial_pose[0] += v * math.cos(robot_initial_pose[2]) * dt
    robot_initial_pose[1] += v * math.sin(robot_initial_pose[2]) * dt
    robot_initial_pose[2] += omega * dt

    slam.predict_particles(v, omega, dt)
    
    print("After motion update:")
    for i, p in enumerate(slam.particles):
        print(f"Particle {i}: x={p.x:.2f}, y={p.y:.2f}, theta={p.theta:.2f}")

    # --- First landmark observation (with noise), observed twice ---
    marker_id_1 = 1
    true_landmark_1 = np.array([2.0, 1.0])

    for obs_num in range(2):
        z_all = []
        dx = true_landmark_1[0] - robot_initial_pose[0]
        dy = true_landmark_1[1] - robot_initial_pose[1]
        range_ = math.hypot(dx, dy) + np.random.normal(0, 0.2)
        bearing = wrap_angle_rad(math.atan2(dy, dx) - robot_initial_pose[2] + np.random.normal(0, np.radians(5)))
        z_all.append([marker_id_1, range_, bearing])
        slam.observation_update(z_all)
        print(f"\nLandmark 1 observation {obs_num+1}:")
        for i, p in enumerate(slam.particles):
            print(f"Particle {i} landmark positions: {p.landmarks_position}")

    # --- Second landmark observation (with noise), observed twice ---
    marker_id_2 = 2
    true_landmark_2 = np.array([1.0, 3.0])

    for obs_num in range(2):
        z_all = []
        dx = true_landmark_2[0] - robot_initial_pose[0]
        dy = true_landmark_2[1] - robot_initial_pose[1]
        range_ = math.hypot(dx, dy) + np.random.normal(0, 0.2)
        bearing = wrap_angle_rad(math.atan2(dy, dx) - robot_initial_pose[2] + np.random.normal(0, np.radians(5)))
        z_all.append([marker_id_2, range_, bearing])
        slam.observation_update(z_all)
        print(f"\nLandmark 2 observation {obs_num+1}:")
        for i, p in enumerate(slam.particles):
            print(f"Particle {i} landmark positions: {p.landmarks_position}")

    marker_id = 1
    particle = slam.particles[0]  # or any particle

    if marker_id in particle.landmarks_id:
        idx = particle.landmarks_id.index(marker_id)
        landmark1_pos = particle.landmarks_position[idx]
        print("Landmark 1 position:", landmark1_pos)
    else:
        print("Landmark 1 not observed by this particle.")
        
    # Estimate final state
    x_est = slam.calc_final_state()
    print("\nEstimated state (weighted mean):", x_est.flatten())

    # Get best particle
    best = slam.get_best_particle()
    print("Best particle pose:", best.x, best.y, best.theta)

def test_resampling():
    print("Testing resampling...")

    # Create FastSLAM with 5 particles
    slam = FastSLAM([0,0,0], 5, (0.1, 0.01), 10, np.eye(2))

    # Assign distinct weights
    for i, p in enumerate(slam.particles):
        p.weight = i + 1  # weights: 1, 2, 3, 4, 5

    # Normalize weights
    slam.normalize_weight()
    print("Weights before resampling:", [p.weight for p in slam.particles])

    # Save original particle IDs (using id() or add a custom attribute)
    original_ids = [id(p) for p in slam.particles]

    # Perform resampling
    slam.resampling()

    # Check which original particles were selected
    new_ids = [id(p) for p in slam.particles]
    print("Original IDs:", original_ids)
    print("New IDs after resampling:", new_ids)
    print("Resampled indexes:", slam.resampled_indexes)

    # Count how many times each original particle was selected
    from collections import Counter
    counts = Counter(slam.resampled_indexes)
    print("Selection counts:", counts)

if __name__ == "__main__":
    test_resampling()

# if __name__ == "__main__":
#     test_fastslam()