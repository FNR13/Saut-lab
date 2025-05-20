import math
import numpy as np

import pygame

def wrap_angle_rad(angle):
    """Wrap angle to [0, 2π)"""
    return angle % (2 * math.pi)

def draw_fastslam_particles(particles, win, color=(0, 0, 255)):
    """Draw all particles in the FastSLAM algorithm."""

    for particle in particles:
        pos = (int(particle.x), int(particle.y))
        pygame.draw.circle(win, (0, 0, 0), pos, 4)  # black outline
        pygame.draw.circle(win, color, pos, 3)      # main colored dot

def draw_covariance_ellipse(win, mean, cov, color=(255, 0, 0), scale=2.0, min_size=5.0):
    eigenvals, eigenvecs = np.linalg.eig(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    angle = math.degrees(math.atan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Compute ellipse axes
    width = max(min_size, 2 * scale * np.sqrt(eigenvals[0]))
    height = max(min_size, 2 * scale * np.sqrt(eigenvals[1]))

    # Draw on a transparent surface
    ellipse_surf = pygame.Surface((width, height), pygame.SRCALPHA)
    pygame.draw.ellipse(ellipse_surf, (*color, 100), (0, 0, width, height))
    
    ellipse_rot = pygame.transform.rotate(ellipse_surf, -angle)
    rect = ellipse_rot.get_rect(center=(mean[0], mean[1]))
    win.blit(ellipse_rot, rect)


