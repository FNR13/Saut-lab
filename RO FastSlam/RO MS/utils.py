import math
import numpy as np
from matplotlib.patches import Ellipse


import pygame

def wrap_angle_rad(angle):
    """Wrap angle to [-π, π)"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def draw_fastslam_particles(particles, win, color=(0, 0, 255)):
    """Draw all particles in the FastSLAM algorithm."""

    for particle in particles:
        pos = (int(particle.x), int(particle.y))
        pygame.draw.circle(win, (0, 0, 0), pos, 4)  # black outline
        pygame.draw.circle(win, color, pos, 3)      # main colored dot

def draw_covariance_ellipse(win, mean, cov, color=(255, 0, 0), scale=2.0, min_size=5.0):
    """Draw covarice ellipses in Microsimulator"""

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

def update_paths(paths, new_pose):
    '''Add the last pose of each particle to its path'''
    return np.concatenate((paths, new_pose), axis=1)

def resample_paths(paths,indexes):
    '''Resample the particles with replacement'''
    # Resampling 
    paths = paths[indexes, :, :]
    return paths

def draw_ellipse(ax, mean, cov, scale=2.0, min_size=5.0, color='0.7'):
    """Draw covarice ellipses in final Map"""

    eigenvals, eigenvecs = np.linalg.eig(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    angle = math.degrees(math.atan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Compute ellipse axes
    width = max(min_size, 2 * scale * np.sqrt(eigenvals[0]))
    height = max(min_size, 2 * scale * np.sqrt(eigenvals[1]))
    
    # Change the coordenate origin  
    mean = mean.copy()
    mean[-1] = -mean[-1]
    #mean[0] = -mean[0]

    # Create the ellipse
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, edgecolor=(0,1,0),facecolor='none')
    ax.add_patch(ellipse)

    # Draw the mean 
    axis_length1 = 0.5 * width
    axis_length2 = 0.5 * height
    dir1 = eigenvecs[:, 0] * axis_length1
    dir2 = eigenvecs[:, 1] * axis_length2

    ax.plot([mean[0] - dir1[0], mean[0] + dir1[0]], 
            [mean[1] - dir1[1], mean[1] + dir1[1]],
            color=color, linestyle='-', linewidth=0.5)

    ax.plot([mean[0] - dir2[0], mean[0] + dir2[0]], 
            [mean[1] - dir2[1], mean[1] + dir2[1]],
            color=color, linestyle='-', linewidth=0.5)
    
    return ellipse #Necessary to write the label in the final map

def rotate_path(x, y, theta):

    # Use the coordinates origin as pivot 
    cx, cy = x[0], y[0]

    # Rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    x_rotated = []
    y_rotated = []

    for xi, yi in zip(x, y):
        # Translate to origin with respect to the first point
        x_tras = xi - cx
        y_tras = yi - cy

        # Apply rotation
        x_rot = cos_theta * x_tras - sin_theta * y_tras
        y_rot = sin_theta * x_tras + cos_theta * y_tras

        # Translate again to the initial point as pivot
        x_final = x_rot + cx
        y_final = y_rot + cy

        x_rotated.append(x_final)
        y_rotated.append(y_final)

    x_rotated = np.array(x_rotated)
    y_rotated = np.array(y_rotated)

    return x_rotated, y_rotated

