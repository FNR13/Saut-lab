import math
import numpy as np

# -----------------------------------------------------------------------------------------------------------------
# Importing EKF
try:
    # Relative import for normal package usage
    from .ekf import ExtendedKalmanFilter, compute_jacobians

except ImportError:
    # Absolute import fallback for direct script testing
    from ekf import ExtendedKalmanFilter, compute_jacobians

# Importing utils
try:
    # Relative import for normal package usage
    from classUtils.utils import wrap_angle_rad

except ModuleNotFoundError:
    # Absolute import fallback for direct script testing
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from classUtils.utils import wrap_angle_rad

# -----------------------------------------------------------------------------------------------------------------
# Class Definition

class Particle:
    def __init__(self, odometry_uncertainty, landmark_uncertainty, Q_cov, sensor_max_range, sensor_min_range):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.weight = 1

        self.sensor_max_range = sensor_max_range
        self.sensor_min_range = sensor_min_range
        self.odometry_noise = odometry_uncertainty
        self.landmark_uncertainty = landmark_uncertainty
        self.Q_cov = Q_cov

        self.landmarks_id                  = []
        self.landmarks_position            = []
        self.landmarks_position_covariance = []
        self.landmarks_observation_count   = []  
        self.landmarks_EKF                 = [] 

    def motion_update(self, v, omega, dt):

        # Add noise to simulate uncertainty
        v += np.random.normal(0, self.odometry_noise[0])
        omega += np.random.normal(0, self.odometry_noise[1])

        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += omega * dt
    
    
    def landmark_update(self, marker_id, z):
        # Check if the landmark is already known
        if marker_id in self.landmarks_id:

            # Get index of the landmark from the list
            idx = self.landmarks_id.index(marker_id) 

            pose = np.array([self.x, self.y, self.theta]).reshape(3, 1)

            self.landmarks_EKF[idx].update(z, pose) 
            self.landmarks_position[idx] = self.landmarks_EKF[idx].landmark_position.reshape(2,1)  # Change the landmark to a 2x1 array
            self.landmarks_position_covariance[idx] = self.landmarks_EKF[idx].landmark_covariances
            self.landmarks_observation_count[idx] += 1
            self.weight *= self.compute_weight(idx, z, pose)

        else:

            # Add new landmark (Initial guess)           
            bearing  = z
            theta = wrap_angle_rad(self.theta + bearing) #Can give problems 
            
            r_min = self.sensor_min_range
            r_max = self.sensor_max_range
            rand_range = np.random.uniform(r_min, r_max)  
          
            landmark_x = self.x + rand_range * math.cos(theta)
            landmark_y = self.y + rand_range * math.sin(theta)

            landmark_Position = np.array([landmark_x, landmark_y]).reshape(2,1)
            landmark_position_covariance = np.array([[self.landmark_uncertainty, 0], [0, self.landmark_uncertainty]])

            newKalmanFilter = ExtendedKalmanFilter(landmark_Position, landmark_position_covariance, self.Q_cov)

            self.landmarks_id.append(marker_id)
            self.landmarks_position.append(landmark_Position)
            self.landmarks_position_covariance.append(landmark_position_covariance)
            self.landmarks_observation_count.append(0)
            self.landmarks_EKF.append(newKalmanFilter)


    def compute_weight(self, idx, z, pose):
            
            xf = self.landmarks_EKF[idx].landmark_position
            Pf = self.landmarks_EKF[idx].landmark_covariances

            zp, _, Hf, Sf = compute_jacobians(pose, xf, Pf, self.Q_cov)

            # Angle offset
            dz = wrap_angle_rad(z - zp)            

            try:
                Sf = float(Sf)      
                inv_Sf = 1.0 / Sf
            except np.linalg.LinAlgError:
                return 1e-8
            
            num = math.exp(-0.5 * dz * inv_Sf * dz)
            den = math.sqrt(2 * math.pi * Sf)
            weight = num / den

            return max(weight, 1e-8)

# -----------------------------------------------------------------------------------------------------------------
# Test function for the Particle class
def test_particle():
    import numpy as np

    # Create a particle with some uncertainty
    odometry_uncertainty = (0.01, 0.01)
    landmark_uncertainty = 100.0
    Q_cov = 1.47227856e-10       
    sensor_max_range = 5.35686663476375
    sensor_min_range = 0.33607217464420264

    p = Particle(odometry_uncertainty, landmark_uncertainty, Q_cov, sensor_max_range, sensor_min_range)

    # Set initial pose
    print("Initial pose:", p.x, p.y, p.theta)

    # Simulate motion update
    p.motion_update(1.0, 0.1, 1.0)
    print("After motion update:", p.x, p.y, p.theta, "\n")

    # Simulate a landmark observation (range, bearing)
    marker_id = 42
    true_landmark = np.array([3.0, 4.0])
    bearing_meas = np.arctan2(true_landmark[1] - p.y, true_landmark[0] - p.x) - p.theta

    # Add noise to the first measurement
    bearing_meas_noisy = bearing_meas + np.random.normal(0, np.deg2rad(5))
    z = bearing_meas_noisy

    # First observation (should initialize the landmark)
    p.landmark_update(marker_id, z)
    print("After first landmark observation (with noise):")
    print("    Landmark IDs:", p.landmarks_id)
    print("    Landmark positions:", p.landmarks_position)
    print("    Landmark covariances:", p.landmarks_position_covariance)
    print("    Weights:", p.weight, "\n")

    # Second observation (use the true measurement, no noise)
    z_true = bearing_meas
    p.landmark_update(marker_id, z_true)
    print("After second landmark observation (true measurement):")
    print("    Landmark positions:", p.landmarks_position)
    print("    Landmark covariances:", p.landmarks_position_covariance)
    print("    Observation count:", p.landmarks_observation_count)
    print("    Weights:", p.weight)

if __name__ == "__main__":
    test_particle()