import math
import numpy as np

from utils import wrap_angle_rad


class ExtendedKalmanFilter:
    def __init__(self, initial_landmark_position, initial_landmark_covariances, Q_cov):

        self.landmark_position   = initial_landmark_position    # np (x,y) positions of the landmark
        self.landmark_covariances = initial_landmark_covariances # np Matrix of covariance of the landmark
        self.Q_cov                = Q_cov               # Matrix of measurement noise covariance for range and bearing

    # There is no predic step in the EKF for landmarks, as landmarks are static

    def update(self, measurement, pose):
        z = measurement.reshape(2, 1)

        zp, Hv, Hf, Sf = compute_jacobians(pose, self.landmark_position, self.landmark_covariances, self.Q_cov)

        # Distance from the linearization point
        dz = z - zp
        print(dz)
        dz[1] = wrap_angle_rad(dz[1])

        xf, Pf = update_kf_with_cholesky(self.landmark_position, self.landmark_covariances, dz, self.Q_cov, Hf)

        self.landmark_position = xf
        self.landmark_covariances = Pf
            
def compute_jacobians(pose, xf, Pf, Q_cov):
    """
    Robot7Particles are the mesurments therfore noise is expected
    Compute expected measurement, Jacobians, and innovation covariance for EKF update.
    
    Parameters:
    - pose: the robot object with x, y, theta 
    - xf: landmark mean position as a 2x1 numpy array [[x], [y]]
    - Pf: 2x2 covariance matrix of the landmark
    - Q_cov: 2x2 measurement noise covariance (sensor noise)

    Returns:
    - zp: 2x1 predicted measurement [range; bearing]
    - Hv: 2x3 Jacobian of the measurement w.r.t. robot state
    - Hf: 2x2 Jacobian of the measurement w.r.t. landmark position
    - Sf: 2x2 innovation covariance matrix
    """
    x = pose[0]
    y = pose[1]
    theta = pose[2]

    # Difference in x and y between landmark and robot
    dx = xf[0] - x
    dy = xf[1] - y

    # Squared and actual distance
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    # Predicted measurement: range and bearing
    zp = np.array([
        d,  # range
        wrap_angle_rad(math.atan2(dy, dx) - theta)  # bearing (angle between robot orientation and landmark)
    ]).reshape(2, 1)

    # Jacobian w.r.t. robot pose [x, y, theta]
    Hv = np.array([
        [-dx / d,     -dy / d,      0.0],
        [dy / d2,     -dx / d2,    -1.0]
    ])

    # Jacobian w.r.t. landmark position [xf, yf]
    Hf = np.array([
        [ dx / d,     dy / d],
        [-dy / d2,    dx / d2]
    ])

    # Innovation covariance matrix (uncertainty in prediction)
    Sf = Hf @ Pf @ Hf.T + Q_cov

    return zp, Hv, Hf, Sf

def update_kf_with_cholesky(xf, Pf, v, Q_cov, Hf):
    """
    Provides numerical Stability when inverting the matrix 
    Perform an Extended Kalman Filter (EKF) update step for a landmark using Cholesky decomposition.

    Parameters:
    - xf: (2x1 np.array) current estimate of the landmark position
    - Pf: (2x2 np.array) covariance matrix of the landmark estimate
    - v: (2x1 np.array) innovation vector (z_actual - z_predicted)
    - Q_cov: (2x2 np.array) measurement noise covariance matrix
    - Hf: (2x2 np.array) Jacobian of the measurement model w.r.t. the landmark position

    Returns:
    - x: (2x1 np.array) updated landmark position estimate
    - P: (2x2 np.array) updated landmark covariance
    """

    PHt = Pf @ Hf.T                      # Cross covariance
    S = Hf @ PHt + Q_cov                 # Innovation covariance

    S = (S + S.T) * 0.5                  # Symmetrize S for numerical stability
    s_chol = np.linalg.cholesky(S).T    # Cholesky decomposition of S
    s_chol_inv = np.linalg.inv(s_chol)  # Inverse of upper Cholesky factor
    W1 = PHt @ s_chol_inv               # Intermediate step for Kalman gain
    W = W1 @ s_chol_inv.T               # Kalman gain

    print(np.shape(v))
    x = xf + W @ v                      # Updated landmark mean
    P = Pf - W1 @ W1.T                  # Updated landmark covariance

    return x, P





        
