import math
import numpy as np

import pygame

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

        if keys[pygame.K_4]:  # Increase left
            self.velL = min(self.velL + increment, self.max)
        if keys[pygame.K_1]:  # Decrease left
            self.velL = max(self.velL - increment, self.min)
        if keys[pygame.K_6]:  # Increase right
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_3]:  # Decrease right
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_5]:  # Increase both
            self.velL = min(self.velL + increment, self.max)
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_2]:  # Decrease both
            self.velL = max(self.velL - increment, self.min)
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_SPACE]:  # Equalize both velocities
            avg = (self.velL + self.velR) / 2
            self.velL = self.velR = avg

    def update_kinematics(self, dt):
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