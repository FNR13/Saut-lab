import pygame

import math
import random


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
    

