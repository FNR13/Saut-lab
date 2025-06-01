import math
import numpy as np
from matplotlib.patches import Ellipse

import pygame

import rosbag

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

def read_bag_data(bag_file):
    bag = rosbag.Bag(bag_file)
    pose_times = []
    pose_vectors = []
    obs_times = []
    obs_data = []

    for topic, msg, t in bag.read_messages(topics=['/pose']):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation_z = msg.pose.pose.orientation.z
        linear_x = msg.twist.twist.linear.x
        angular_z = - msg.twist.twist.angular.z
        pose_vectors.append([t.to_sec(), x, y, orientation_z, linear_x, angular_z])
        pose_times.append(t.to_sec())

    for topic, msg, t in bag.read_messages(topics=['/fiducial_transforms']):
        observations = []
        for transform in msg.transforms:
            fiducial_id = transform.fiducial_id
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            observations.append([fiducial_id, tx, ty, tz])
        obs_data.append([t.to_sec(), observations])
        obs_times.append(t.to_sec())

    bag.close()

    # Align time to start at zero
    time_bias = min(pose_times[0], obs_times[0])
    pose_times = [t - time_bias for t in pose_times]
    obs_times = [t - time_bias for t in obs_times]
    for entry in obs_data:
        entry[0] -= time_bias
    for entry in pose_vectors:
        entry[0] -= time_bias

    pose_vectors = np.array(pose_vectors)

    # Interpolate pose values at obs_times
    interp_x = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 1])
    interp_y = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 2])
    interp_orientation_z = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 3])
    interp_linear_x = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 4])
    interp_angular_z = np.interp(obs_times, pose_vectors[:, 0], pose_vectors[:, 5])

    return obs_times, interp_x, interp_y, interp_orientation_z, interp_linear_x, interp_angular_z, obs_data


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


