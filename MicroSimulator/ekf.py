import math
import numpy as np

import pygame

from utils import wrap_angle_rad

def compute_jacobians(robot, xf, Pf, Q_cov):
    """
    Robot7Particles are the mesurments therfore noise is expected
    Compute expected measurement, Jacobians, and innovation covariance for EKF update.
    
    Parameters:
    - robot: the robot object with x, y, theta (pose)
    - xf: landmark mean position as a 2x1 numpy array [[x], [y]]
    - Pf: 2x2 covariance matrix of the landmark
    - Q_cov: 2x2 measurement noise covariance (sensor noise)

    Returns:
    - zp: 2x1 predicted measurement [range; bearing]
    - Hv: 2x3 Jacobian of the measurement w.r.t. robot state
    - Hf: 2x2 Jacobian of the measurement w.r.t. landmark position
    - Sf: 2x2 innovation covariance matrix
    """

    # Difference in x and y between landmark and robot
    dx = xf[0, 0] - robot.x
    dy = xf[1, 0] - robot.y

    # Squared and actual distance
    d2 = dx ** 2 + dy ** 2
    d = math.sqrt(d2)

    # Predicted measurement: range and bearing
    zp = np.array([
        d,  # range
        wrap_angle_rad(math.atan2(dy, dx) - robot.theta)  # bearing (angle between robot orientation and landmark)
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

    x = xf + W @ v                      # Updated landmark mean
    P = Pf - W1 @ W1.T                  # Updated landmark covariance

    return x, P

def update_landmark(robot, z, Q_cov):
    lm_id = int(z[2])
    xf = np.array(robot.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(robot.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, Hv, Hf, Sf = compute_jacobians(robot, xf, Pf, Q_cov)

    dz = z[0:2].reshape(2, 1) - zp
    dz[1, 0] = wrap_angle_rad(dz[1, 0])

    xf, Pf = update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)

    robot.lm[lm_id, :] = xf.T
    robot.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

    return robot

def draw_covariance_ellipse(win, mean, cov, color=(255, 0, 0), scale=2.0):
    eigenvals, eigenvecs = np.linalg.eig(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    angle = math.degrees(math.atan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * scale * np.sqrt(eigenvals)

    ellipse_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.ellipse(ellipse_surf, (*color, 100), (0, 0, width, height))
    ellipse_rot = pygame.transform.rotate(ellipse_surf, -angle)
    rect = ellipse_rot.get_rect(center=(mean[0], mean[1]))
    win.blit(ellipse_rot, rect) 

        
