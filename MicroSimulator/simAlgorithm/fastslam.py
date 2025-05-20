import math
import numpy as np

from simAlgorithm.particle import Particle
# from particle import Particle # For debugging

from utils import wrap_angle_rad


class FastSLAM:
    def __init__(self, robot_initial_pose, num_particles, particles_odometry_uncertanty, landmarks_initial_uncertenty, Q_cov):

        self.robot_initial_pose = robot_initial_pose  # (x, y, theta)

        # Particle Filter parameters
        self.num_particles = num_particles

        self.particles = self._init_particles(
            particles_odometry_uncertanty, landmarks_initial_uncertenty, Q_cov)

    # Particle Filter Management
    def _init_particles(self,particles_odometry_uncertanty, landmarks_initial_uncertenty, Q_cov):
        return [
            Particle(particles_odometry_uncertanty, landmarks_initial_uncertenty, Q_cov)
            for _ in range(self.num_particles)
        ]

    def predict_particles(self, v, omega, dt):
        for particle in self.particles:
            particle.motion_update(v, omega, dt)
    
    # Landmark Management
    def observation_update(self, z_all):

        for observation in z_all:

            marker_id = observation[0]
            z = np.array(observation[1:3]).reshape(2, 1)

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


def test_fastslam():
    import numpy as np

    # Initial robot pose
    robot_initial_pose = [0.0, 0.0, 0.0]
    num_particles = 3
    particles_odometry_uncertanty = (0.1, 0.01)  # (speed, angular rate)
    landmarks_initial_uncertanty = 10  # Initial uncertainty for landmarks
    Q_cov = np.diag([20.0, np.radians(30)])  # Measurement noise for fast slam - for range and bearing

    # Create FastSLAM object
    slam = FastSLAM(robot_initial_pose, num_particles, particles_odometry_uncertanty, landmarks_initial_uncertanty, Q_cov)

    # Simulate motion update
    v = 1
    omega = 0
    dt = 1.0

    robot_initial_pose[0] += v * math.cos(robot_initial_pose[2]) * dt
    robot_initial_pose[1] += v * math.sin(robot_initial_pose[2])* dt
    robot_initial_pose[1] += omega * dt

    slam.predict_particles(v, omega, dt)
    
    print("After motion update:")
    for i, p in enumerate(slam.particles):
        print(f"Particle {i}: x={p.x:.2f}, y={p.y:.2f}, theta={p.theta:.2f}")

    # First landmark observation (with noise)
    marker_id = 1
    true_landmark = np.array([2.0, 1.0])
    z_all = []

    # Real system camera feed
    dx = true_landmark[0] - robot_initial_pose[0]
    dy = true_landmark[1] - robot_initial_pose[1]

    range = math.hypot(dx, dy)
    bearing = wrap_angle_rad(math.atan2(dy, dx) - robot_initial_pose[2])

    z_all.append([marker_id, range, bearing])
    slam.observation_update(z_all)

    print("\nFirst landmark observation (using observation_update):")
    for i, p in enumerate(slam.particles):
        print(f"Particle {i} landmark positions: {p.landmarks_position}")

    # Second landmark observation (with new noise)
    z_all = []

    # Real system camera feed
    dx = true_landmark[0] - robot_initial_pose[0]
    dy = true_landmark[1] - robot_initial_pose[1]

    range = math.hypot(dx, dy)
    bearing = wrap_angle_rad(math.atan2(dy, dx) - robot_initial_pose[2])

    z_all.append([marker_id, range, bearing])
    slam.observation_update(z_all)

    print("\nSecond landmark observation (using observation_update):")
    for i, p in enumerate(slam.particles):
        print(f"Particle {i} landmark positions: {p.landmarks_position}")

    # Estimate final state
    x_est = slam.calc_final_state()
    print("\nEstimated state (weighted mean):", x_est.flatten())

    # Get best particle
    best = slam.get_best_particle()
    print("Best particle pose:", best.x, best.y, best.theta)

if __name__ == "__main__":
    test_fastslam()