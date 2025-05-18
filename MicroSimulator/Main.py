import math
import pygame
import random
import numpy as np


def wrap_angle_rad(angle):
    """Wrap angle to [0, 2π)"""
    return angle % (2 * math.pi)

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


class Envo:
    def __init__(self,dimensions):
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.green = (0,255,0)
        self.blue = (0,0,255)
        self.red = (255,0,0)
        self.yellow = (255,255,0)

        self.length = dimensions[0]
        self.width = dimensions[1]

        pygame.display.set_caption("Differential Drive Robot")
        self.win = pygame.display.set_mode((self.length,self.width))

        self.font = pygame.font.Font('freesansbold.ttf',35)
        self.text = self.font.render('default',True,self.white,self.black)
        self.textrect = self.text.get_rect()
        self.textrect.center=(dimensions[1]-175, dimensions[0]-500)
        self.traj_set=[]

    def write (self, velL, velR, theta):
        txt = f"Vel L = {velL/3779.52:.3f} Vel R ={velR/3779.52:.3f} theta = {int(math.degrees(wrap_angle_rad(theta)))}"
        self.text=self.font.render(txt,True,self.white,self.black)
        self.win.blit(self.text,self.textrect)

    def trajectory(self,pos):
        for i in range(0,len(self.traj_set)-1):
            pygame.draw.line(self.win,self.red,(self.traj_set[i][0],self.traj_set[i][1]),(self.traj_set[i+1][0],self.traj_set[i+1][1]))
        if self.traj_set.__sizeof__()>300000:
            self.traj_set.pop(0)
        self.traj_set.append(pos)


class Robot:
    def __init__(self, startpoint, robotimg, width, n_landmarks=50):
        self.met2pix = 3779.52
        self.wd = width
        self.x = startpoint[0]
        self.y = startpoint[1]
        self.theta = 0
        self.velL = 0.01 * self.met2pix
        self.velR = 0.01 * self.met2pix
        self.max = 0.05 * self.met2pix
        self.min = 0.002 * self.met2pix

        # --- Mapping ---
        self.lm = np.full((n_landmarks, 2), np.nan)
        self.lmP = np.full((2 * n_landmarks, 2), np.nan)
        self.observed_landmarks = {}  # mapping from landmark_id to index
        self.lm_observation_count = np.zeros(n_landmarks, dtype=int)

        # --- Graphics ---
        self.imge = pygame.image.load(robotimg).convert_alpha()
        self.rotate = self.imge
        self.rect = self.rotate.get_rect(center=(self.x, self.y))

    def draw(self,win):
        win.blit(self.rotate,self.rect)

    def update_velocities(self, keys):
        increment = 0.0001 * self.met2pix

        if keys[pygame.K_KP4]:  # Increase left
            self.velL = min(self.velL + increment, self.max)
        if keys[pygame.K_KP1]:  # Decrease left
            self.velL = max(self.velL - increment, self.min)
        if keys[pygame.K_KP6]:  # Increase right
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_KP3]:  # Decrease right
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_KP5]:  # Increase both
            self.velL = min(self.velL + increment, self.max)
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_KP2]:  # Decrease both
            self.velL = max(self.velL - increment, self.min)
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_SPACE]:  # Equalize both velocities
            avg = (self.velL + self.velR) / 2
            self.velL = self.velR = avg

    def update_kinematics(self):
        self.x += ((self.velL + self.velR) / 2) * math.cos(self.theta) * dt
        self.y -= ((self.velL + self.velR) / 2) * math.sin(self.theta) * dt
        self.theta += (self.velR - self.velL) / self.wd * dt
    
        self.rotate = pygame.transform.rotozoom(self.imge, math.degrees(self.theta), 1)
        self.rect = self.rotate.get_rect(center=(self.x, self.y))
        
    def get_noisy_pose(self, noise_std=(1.0, 1.0, 0.05)):
        """
        Returns the pose (x, y, theta) with added Gaussian noise.
        Parameters:
            noise_std (tuple): Standard deviations for (x, y, theta) noise.
        Returns:
            tuple: (x_noisy, y_noisy, theta_noisy)
        """
        x_noisy = self.x + np.random.normal(0, noise_std[0])
        y_noisy = self.y + np.random.normal(0, noise_std[1])
        theta_noisy = self.theta + np.random.normal(0, noise_std[2])
        return x_noisy, y_noisy, theta_noisy
        

class CarSensor:
    def __init__(self, car_width, sensor_width, sensor_reach, color=(0, 255, 0)):
        self.car_width = car_width
        self.sensor_width = sensor_width
        self.sensor_reach = sensor_reach
        self.color = color

    def compute_trapezoid(self, x, y, theta):
        # Half widths
        half_car = self.car_width / 2
        half_sensor = self.sensor_width / 2

        # Sensor front points
        front_left = (
            x + self.sensor_reach * math.cos(theta) - half_sensor * math.sin(theta),
            y - self.sensor_reach * math.sin(theta) - half_sensor * math.cos(theta),
        )
        front_right = (
            x + self.sensor_reach * math.cos(theta) + half_sensor * math.sin(theta),
            y - self.sensor_reach * math.sin(theta) + half_sensor * math.cos(theta),
        )

        # Sensor rear points (near robot front)
        rear_left = (
            x - half_car * math.sin(theta),
            y - half_car * math.cos(theta),
        )
        rear_right = (
            x + half_car * math.sin(theta),
            y + half_car * math.cos(theta),
        )

        # Return points in order (rear_left, front_left, front_right, rear_right)
        return [rear_left, front_left, front_right, rear_right]

    def point_in_polygon(self, point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and \
                (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
                inside = not inside
        return inside

    def filter_landmarks(self, landmarks_positions, robot_pos, robot_theta):
        polygon = self.compute_trapezoid(robot_pos[0], robot_pos[1], robot_theta)
        visible = []
        for pos in landmarks_positions:
            if self.point_in_polygon(pos, polygon):
                visible.append(pos)
        return visible

    def draw(self, win, robot_pos, robot_theta):
        polygon = self.compute_trapezoid(robot_pos[0], robot_pos[1], robot_theta)
        pygame.draw.polygon(
            win, self.color,
            [(int(p[0]), int(p[1])) for p in polygon], 2
        )

        
class Landmarks:
    def __init__(self, num_landmarks, window_size, radius=8, color=(0, 0, 255), min_dist=100):
        """
        num_landmarks: how many landmarks to generate
        window_size: tuple (width, height) for placing landmarks inside screen
        radius: radius of each landmark circle
        color: RGB color tuple for landmarks
        min_dist: minimum allowed distance between landmarks
        """
        self.num = num_landmarks
        self.width, self.height = window_size
        self.radius = radius
        self.color = color
        self.min_dist = min_dist
        
        self.positions = []
        self._generate_positions()
    
    def _generate_positions(self):

        attempts_limit = 1000  # max tries to place a landmark
        for _ in range(self.num):
            attempts = 0
            while attempts < attempts_limit:
                x = random.randint(self.radius, self.width - self.radius)
                y = random.randint(self.radius, self.height - self.radius)
                pos = (x, y)
                
                if self._is_far_enough(pos):
                    self.positions.append(pos)
                    break
                attempts += 1
            else:
                print(f"Warning: Could only place {_} landmarks out of {self.num} due to spacing constraints.")
                break

    def _is_far_enough(self, pos):
        for p in self.positions:
            dist = math.hypot(pos[0] - p[0], pos[1] - p[1])
            if dist < self.min_dist:
                return False
        return True
    
    def draw(self, win):
        for pos in self.positions:
            pygame.draw.circle(win, self.color, pos, self.radius)
    
    def get_positions(self):
        return self.positions
    
    def get_id_and_positions(self, noise_std=0):
        """
        Returns a list of (id, noisy_position) tuples.
        
        Parameters:
            noise_std (float or tuple): Standard deviation for Gaussian noise.
                                        If float, applies equally to x and y.
                                        If tuple, expected as (std_x, std_y).
        """
        if isinstance(noise_std, (int, float)):
            std_x = std_y = noise_std
        else:
            std_x, std_y = noise_std
    
        noisy_positions = []
        for i, (x, y) in enumerate(self.positions):
            noisy_x = x + np.random.normal(0, std_x)
            noisy_y = y + np.random.normal(0, std_y)
            noisy_positions.append((i, (noisy_x, noisy_y)))
        
        return noisy_positions

        

pygame.init()

dim = (1200, 800)
env = Envo(dim)
start = (400, 200)
rob = Robot(start, "Robot.png", 0.01 * 3779.52)
landmarks = Landmarks(num_landmarks=10, window_size=(dim[0], dim[1]))

sensor = CarSensor(
    car_width=rob.wd,  # your robot width
    sensor_width=200,  # sensor trapezoid width in pixels, tweak as needed
    sensor_reach=150,  # sensor reach in pixels, tweak as needed
    color=(0, 255, 0)  # green trapezoid
)

clock = pygame.time.Clock()
highlighted = set()

dt = 0
prevtime = pygame.time.get_ticks()

run = True

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()
    rob.update_velocities(keys)
    rob.update_kinematics()

    dt = clock.tick(60) / 1000.0
    prevtime = pygame.time.get_ticks()

    env.win.fill(env.white)
    landmarks.draw(env.win)
    rob.draw(env.win)
    env.trajectory((rob.x, rob.y))
    env.write(int(rob.velL), int(rob.velR), rob.theta)

    sensor.draw(env.win, (rob.x, rob.y), rob.theta)

    visible_landmarks = sensor.filter_landmarks(landmarks.get_positions(), (rob.x, rob.y), rob.theta)
    Q_cov = np.diag([20.0, np.radians(30)])

    for idx, pos in enumerate(landmarks.get_positions()):
        # Draw true landmark position as red circle for ground truth
        pygame.draw.circle(env.win, (255, 0, 0), (int(pos[0]), int(pos[1])), 4)

        if pos in visible_landmarks:
            dx = pos[0] - rob.x
            dy = pos[1] - rob.y
            rng = math.hypot(dx, dy)
            brg = wrap_angle_rad(math.atan2(dy, dx) - rob.theta)
            z = np.array([rng, brg, idx])

            if np.isnan(rob.lm[idx, 0]):
                lx = rob.x + rng * math.cos(brg + rob.theta)
                ly = rob.y + rng * math.sin(brg + rob.theta)
                rob.lm[idx, :] = [lx, ly]
                rob.lmP[2 * idx:2 * idx + 2, :] = np.eye(2) * 100.0
                rob.lm_observation_count[idx] = 1  # Initialize count
            else:
                rob = update_landmark(rob, z, Q_cov)
                rob.lm_observation_count[idx] += 1

    for i in range(len(rob.lm)):
        if not np.isnan(rob.lm[i, 0]):
            mean = rob.lm[i, :]
            cov = rob.lmP[2 * i:2 * i + 2, :]

            # Compute ellipse confidence for color
            uncertainty = np.mean(np.diag(cov[:2, :2]))
            if uncertainty < 30:
                color = (0, 255, 0)  # Green = confident
            elif uncertainty < 80:
                color = (255, 165, 0)  # Orange = medium
            else:
                color = (255, 0, 0)  # Red = uncertain

            draw_covariance_ellipse(env.win, mean, cov, color=color)

            # Optional: show uncertainty as text
            font = pygame.font.SysFont(None, 16)
            txt = font.render(f"{uncertainty:.1f}", True, (0, 0, 0))
            env.win.blit(txt, (mean[0] + 5, mean[1] - 5))

            # Console log (optional)
            print(f"Landmark {i}: Obs={rob.lm_observation_count[i]} | Cov={np.diag(cov[:2, :2])}")
    
    pygame.display.update()

pygame.quit()
