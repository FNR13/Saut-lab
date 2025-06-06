import math
import numpy as np

# -----------------------------------------------------------------------------------------------------------------
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

class ExtendedKalmanFilter:
    def __init__(self, initial_landmark_position, initial_landmark_covariances, Q_cov):

        self.landmark_position    = initial_landmark_position.reshape(2,1)     # np (x,y) positions of the landmark
        self.landmark_covariances = initial_landmark_covariances.reshape(2,2); # np Matrix of covariance of the landmark
        self.Q_cov                = Q_cov               # Matrix of measurement noise covariance for bearing

    # There is no predic step in the EKF for landmarks, as landmarks are static

    def update(self, measurement, pose):
        self.landmark_position = self.landmark_position.reshape(2, 1)
        z = measurement

        zp, Hv, Hf, Sf = compute_jacobians(pose, self.landmark_position, self.landmark_covariances, self.Q_cov)

        # Angle offset
        dz = wrap_angle_rad(z - zp)

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
    - Q_cov: measurement bearing noise covariance (sensor noise)

    Returns:
    - zp: predicted measurement [bearing]
    - Hv: 1x3 Jacobian of the measurement w.r.t. robot state
    - Hf: 1x2 Jacobian of the measurement w.r.t. landmark position
    - Sf: innovation variance
    """
    x = pose[0]
    y = pose[1]
    theta = pose[2]

    # Difference in x and y between landmark and robot
    dx = xf[0] - x
    dy = xf[1] - y

    # Conversion from array of a unique float to single float
    dx = float(dx)
    dy = float(dy)
    theta = float(theta)
   
    # Squared and actual distance
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    # Predicted measurement: bearing 
    zp = wrap_angle_rad(math.atan2(dy, dx) - theta) #Is it plus or minus?

    # Jacobian w.r.t. robot pose [x, y, theta]
    Hv = np.array([
        [dy / d2,     -dx / d2,    -1.0]
    ])

    # Jacobian w.r.t. landmark position [xf, yf]
    Hf = np.array([
        [-dy / d2,    dx / d2]
    ])

    # Innovation variance matrix (uncertainty in prediction)
    Sf = Hf @ Pf @ Hf.T + Q_cov

    return zp, Hv, Hf, Sf

def update_kf_with_cholesky(xf, Pf, v, Q_cov, Hf):
    """
    Provides numerical Stability when inverting the matrix 
    Perform an Extended Kalman Filter (EKF) update step for a landmark using Cholesky decomposition.

    Parameters:
     - xf: (2x1 np.array) current estimate of the landmark position
    - Pf: (2x2 np.array) covariance matrix of the landmark estimate
    - v: innovation differential (z_actual - z_predicted)
    - Q_cov: measurement noise varinace
    - Hf: (1x2 np.array) Jacobian of the measurement model w.r.t. the landmark position

    Returns:
    - x: (2x1 np.array) updated landmark position estimate
    - P: (2x2 np.array) updated landmark covariance
    """

    PHt = Pf @ Hf.T                     # Cross covariance
    S = (Hf @ PHt) + Q_cov              # Innovation covariance

    S = float(S)            
    
    W = PHt / S                         # Kalman gain

    x = xf + W * v                      # Updated landmark mean
    P = Pf - W @ Hf @ Pf                # Updated landmark covariance

    return x, P

# -----------------------------------------------------------------------------------------------------------------
# Test function for the Extended Kalman Filter

def test_ekf():
    import numpy as np

    # True landmark position (used to generate the measurement)
    true_landmark_position = np.array([2.0, 3.0])

    # Initial guess for the landmark position (intentionally different)
    initial_landmark_position = np.array([1.5, 2.5])

    initial_landmark_covariances = np.eye(2) * 1.0
    Q_cov = 1.47227856e-10

    # Create EKF object with the initial guess
    ekf = ExtendedKalmanFilter(initial_landmark_position, initial_landmark_covariances, Q_cov)

    # Simulated robot pose (x, y, theta)
    pose = np.array([1.0, 2.0, np.deg2rad(30)])

    print("Before update:")
    print("Landmark position:", ekf.landmark_position, '\n')
    print("Landmark covariance:\n", ekf.landmark_covariances)

    # Simulated measurement: range from robot to the TRUE landmark
    measurement = np.linalg.norm(true_landmark_position - pose[:2])


    ekf.update(measurement, pose)

    print("\nAfter update:")
    print("Landmark position:", ekf.landmark_position)
    print("Landmark covariance:\n", ekf.landmark_covariances)

    measurement = np.linalg.norm(true_landmark_position - pose[:2])
      

    ekf.update(measurement, pose)

    print("\nAfter  2nd update:")
    print("Landmark position:", ekf.landmark_position)
    print("Landmark covariance:\n", ekf.landmark_covariances)

if __name__ == "__main__":
    test_ekf()       
