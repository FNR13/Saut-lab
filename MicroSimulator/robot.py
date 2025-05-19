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

        if keys[pygame.K_RIGHT]:  # Increase left
            self.velL = min(self.velL + increment, self.max)
        if keys[pygame.K_LEFT]:  # Decrease left
            self.velL = max(self.velL - increment, self.min)
        if keys[pygame.K_LEFT]:  # Increase right
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_RIGHT]:  # Decrease right
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_UP]:  # Increase both
            self.velL = min(self.velL + increment, self.max)
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_DOWN]:  # Decrease both
            self.velL = max(self.velL - increment, self.min)
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_SPACE]:  # Equalize both velocities
            avg = (self.velL + self.velR) / 2
            self.velL = self.velR = avg


    def update_kinematics(self, dt):
        self.x += ((self.velL + self.velR) / 2) * math.cos(self.theta) * dt
        self.y -= ((self.velL + self.velR) / 2) * math.sin(self.theta) * dt
        self.theta += (self.velR - self.velL) / self.wd * dt

        # v += np.random.normal(0, noise_std[0])
        # omega += np.random.normal(0, noise_std[2])

        self.rotate = pygame.transform.rotozoom(self.imge, math.degrees(self.theta), 1)
        self.rect = self.rotate.get_rect(center=(self.x, self.y))
        
        
class Particle:
    def __init__(self, x, y, theta, weight=1.0, robot_width=0.01 * 3779.52, n_landmarks=50,
                 noise_std=(2.0, 2.0, 0.05)):  # std dev for x, y (pixels), theta (radians)
        self.x = x 
        self.y = y
        self.theta = theta
        self.weight = weight
        self.wd = robot_width
        self.lm = np.full((n_landmarks, 2), np.nan)
        self.lmP = np.full((2 * n_landmarks, 2), np.nan)
        self.lm_observation_count = np.zeros(n_landmarks, dtype=int)

    def motion_update(self, velL, velR, dt, noise_std=(1.0, 1.0, 0.02)):
        v = (velL + velR) / 2.0
        omega = (velR - velL) / self.wd
    
        # Add noise to simulate uncertainty
        v += np.random.normal(0, noise_std[0])
        omega += np.random.normal(0, noise_std[2])
    
        self.x += v * math.cos(self.theta) * dt
        self.y -= v * math.sin(self.theta) * dt
        self.theta += omega * dt


    def draw(self, win, color=(150, 150, 255)):
        pos = (int(self.x), int(self.y))
        pygame.draw.circle(win, (0, 0, 0), pos, 4)  # black outline
        pygame.draw.circle(win, color, pos, 3)      # main colored dot

