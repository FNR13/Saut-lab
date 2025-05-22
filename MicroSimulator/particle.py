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
        self.y -= v * math.sin(self.theta) * dt
        self.theta += omega * dt
    
    
    def landmark_update(self, marker_id, z):
        # Check if the landmark is already known
        if marker_id in self.landmarks_id:

            # get index of the landmark from the list
            idx = self.landmarks_id.index(marker_id) #Does it have sense? Is not the marker id the same as the idx?

            pose = np.array([self.x, self.y, self.theta]).reshape(3, 1)

            self.landmarks_EKF[idx].update(z, pose) #PUEDE SER QUE ESTE METIENDO LA POSE DEL ROBOT Y NO EL INDICADOR PARA SABER LA MEDIA
            self.landmarks_position[idx] = self.landmarks_EKF[idx].landmark_position
            self.landmarks_position_covariance[idx] = self.landmarks_EKF[idx].landmark_covariances
            self.landmarks_observation_count[idx] += 1
            self.weight *= self.compute_weight(marker_id, z, pose)

        else:
            # Add new landmark            
            range = z[0]
            bearing = z[1]

            s = math.sin(self.theta + bearing)
            c = math.cos(self.theta + bearing)

            landmark_x = self.x + range * c
            landmark_y = self.y - range * s
            
            landmark_Position = np.array([landmark_x, landmark_y])
            landmark_position_covariance = np.array([[self.landmark_uncertanty, 0], [0, self.landmark_uncertanty]])

            newKalmanFilter = ExtendedKalmanFilter(landmark_Position, landmark_position_covariance, self.Q_cov)

            self.landmarks_id.append(marker_id)
            self.landmarks_position.append(landmark_Position)
            self.landmarks_position_covariance.append(landmark_position_covariance)
            self.landmarks_observation_count.append(0)
            self.landmarks_EKF.append(newKalmanFilter)
            

    def compute_weight(self, marker_id, z, pose):
            
            idx = self.landmarks_id.index(marker_id)
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




