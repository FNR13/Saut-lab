import pygame

import math

class Robot:
    def __init__(self, startpoint, robotimg, width, n_landmarks=50):
        self.met2pix = 3779.52 # Pixels to m/s conversion
        self.wd = width
        self.x = startpoint[0]
        self.y = startpoint[1]
        self.theta =  startpoint[2]
        self.velL = 0.01 * self.met2pix
        self.velR = 0.01 * self.met2pix
        self.max = 0.05 * self.met2pix
        self.min = 0.002 * self.met2pix

            # --- Graphics ---
        self.imge = pygame.image.load(robotimg).convert_alpha()
        self.rotate = self.imge
        self.rect = self.rotate.get_rect(center=(self.x, self.y))

        # # --- Mapping ---
        # self.lm = np.full((n_landmarks, 2), np.nan)
        # self.lmP = np.full((2 * n_landmarks, 2), np.nan)
        # self.observed_landmarks = {}  # mapping from landmark_id to index
        # self.lm_observation_count = np.zeros(n_landmarks, dtype=int)


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

        self.rotate = pygame.transform.rotozoom(self.imge, math.degrees(self.theta), 1)
        self.rect = self.rotate.get_rect(center=(self.x, self.y))
        
        
