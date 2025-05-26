import math
import numpy as np

from ekf import ExtendedKalmanFilter, compute_jacobians
from utils import wrap_angle_rad

class Particle:
    def __init__(self, odometry_uncertanty, landmark_uncertanty, Q_cov):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.weight = 1

        self.odometry_noise = odometry_uncertanty
        self.landmark_uncertanty = landmark_uncertanty
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

            # get index of the landmark from the list
            idx = self.landmarks_id.index(marker_id) 

            pose = np.array([self.x, self.y, self.theta]).reshape(3, 1)

            self.landmarks_EKF[idx].update(z, pose) 
            self.landmarks_position[idx] = self.landmarks_EKF[idx].landmark_position  # Change the landmark to a 1x2 array
            self.landmarks_position_covariance[idx] = self.landmarks_EKF[idx].landmark_covariances
            self.landmarks_observation_count[idx] += 1
            self.weight *= self.compute_weight(idx, z, pose)

        else:
            # Add new landmark            
            range = z[0]
            bearing = z[1]

            s = math.sin(self.theta + bearing)
            c = math.cos(self.theta + bearing)

            landmark_x = self.x + range * c
            landmark_y = self.y + range * s
            
            landmark_Position = np.array([landmark_x, landmark_y]) 
            landmark_position_covariance = np.array([[self.landmark_uncertanty, 0], [0, self.landmark_uncertanty]])

            newKalmanFilter = ExtendedKalmanFilter(landmark_Position, landmark_position_covariance, self.Q_cov)

            self.landmarks_id.append(marker_id)
            self.landmarks_position.append(landmark_Position)
            self.landmarks_position_covariance.append(landmark_position_covariance)
            self.landmarks_observation_count.append(0)
            self.landmarks_EKF.append(newKalmanFilter)
            

    def compute_weight(self, idx, z, pose):
            
            z = z.reshape(2, 1)

            xf = self.landmarks_EKF[idx].landmark_position
            Pf = self.landmarks_EKF[idx].landmark_covariances

            zp, _, Hf, Sf = compute_jacobians(pose, xf, Pf, self.Q_cov)

            # Distance from the linearization point
            dz = z - zp
            dz[1, 0] = wrap_angle_rad(dz[1, 0])

            try:
                inv_Sf = np.linalg.inv(Sf)
            except np.linalg.LinAlgError:
                return 1e-8
            
            num = np.exp(-0.5 * dz.T @ inv_Sf @ dz)[0, 0]
            den = 2.0 * math.pi * np.sqrt(np.linalg.det(Sf))
            weight = num / den

            return max(weight, 1e-8)


def test_particle():
    import numpy as np
    from ekf import ExtendedKalmanFilter

    # Create a particle with some uncertainty
    odometry_uncertanty = (0.01, 0.01)
    landmark_uncertanty = 100.0
    Q_cov = np.diag([0.5, np.deg2rad(10)])

    p = Particle(odometry_uncertanty, landmark_uncertanty, Q_cov)

    # Set initial pose
    print("Initial pose:", p.x, p.y, p.theta)

    # Simulate motion update
    p.motion_update(1.0, 0.1, 1.0)
    print("After motion update:", p.x, p.y, p.theta, "\n")

    # Simulate a landmark observation (range, bearing)
    marker_id = 42
    true_landmark = np.array([3.0, 4.0])
    range_meas = np.linalg.norm(true_landmark - np.array([p.x, p.y]))
    bearing_meas = np.arctan2(true_landmark[1] - p.y, true_landmark[0] - p.x) - p.theta

    # Add noise to the first measurement
    range_meas_noisy = range_meas + np.random.normal(0, 0.2)
    bearing_meas_noisy = bearing_meas + np.random.normal(0, np.deg2rad(5))
    z = np.array([range_meas_noisy, bearing_meas_noisy])

    # First observation (should initialize the landmark)
    p.landmark_update(marker_id, z)
    print("After first landmark observation (with noise):")
    print("    Landmark IDs:", p.landmarks_id)
    print("    Landmark positions:", p.landmarks_position)
    print("    Landmark covariances:", p.landmarks_position_covariance)
    print("    Weights:", p.weight, "\n")

    # Second observation (use the true measurement, no noise)
    z_true = np.array([range_meas, bearing_meas])
    p.landmark_update(marker_id, z_true)
    print("After second landmark observation (true measurement):")
    print("    Landmark positions:", p.landmarks_position)
    print("    Landmark covariances:", p.landmarks_position_covariance)
    print("    Observation count:", p.landmarks_observation_count)
    print("    Weights:", p.weight)

if __name__ == "__main__":
    test_particle()



