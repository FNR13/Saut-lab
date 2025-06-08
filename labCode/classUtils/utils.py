import math
import numpy as np
from matplotlib.patches import Ellipse

# -----------------------------------------------------------------------------------------------------------------
# Debugging and utility functions

def wrap_angle_rad(angle):
    '''Wrap angle to [-π, π)'''
    return (angle + math.pi) % (2 * math.pi) - math.pi

def transform_landmarks(estimated, real):
    '''Align estimated landmarks to real landmarks using orthogonal Procrustes.'''
    
    from scipy.linalg import orthogonal_procrustes

    real = np.array([b.flatten() for b in real])
    estimated = np.array([b.flatten() for b in estimated])
    
    # Center both sets
    real_center = real.mean(axis=0)
    est_center = estimated.mean(axis=0)

    real_centered = real - real_center
    estimated_centered = estimated - est_center
        
    # Find best rotation
    R, _ = orthogonal_procrustes(estimated_centered, real_centered)

    # Apply rotation to estimated
    estimated_rotated = estimated_centered @ R

    # Translate to the real coordenates origin 
    estimated_aligned = estimated_rotated + real_center

    return estimated_aligned, est_center,real_center, R

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


def update_paths(paths, new_pose):
    '''Add the last pose of each particle to its path'''
    return np.concatenate((paths, new_pose), axis=1)

def resample_paths(paths,indexes):
    '''Resample the particles with replacement'''
    # Resampling 
    paths = paths[indexes, :, :]
    return paths

def calculate_NEES(estimated_states, true_states, covariances):
    """
    Compute NEES for each time step.
    """
    T = estimated_states.shape[0]
    nees_values = np.zeros(T)

    for t in range(T):
        error = estimated_states[t] - true_states[t]
        P = covariances[t]
        try:
            nees = error.T @ np.linalg.inv(P) @ error
            nees_values[t] = nees
        except np.linalg.LinAlgError:
            nees_values[t] = np.nan  # mark failure due to singular matrix

    average_nees = np.nanmean(nees_values)
    return nees_values, average_nees    

# -----------------------------------------------------------------------------------------------------------------
# Draw in simulation
def draw_fastslam_particles(particles, win, color=(0, 0, 255)):
    '''Draw all particles in the FastSLAM algorithm.'''
    import pygame

    for particle in particles:
        pos = (int(particle.x), int(particle.y))
        pygame.draw.circle(win, (0, 0, 0), pos, 4)  # black outline
        pygame.draw.circle(win, color, pos, 3)      # main colored dot

def draw_covariance_ellipse(win, mean, cov, color=(255, 0, 0), scale=1.0, min_size=0):
    '''Draw covarice ellipses in Microsimulator'''
    import pygame

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

def draw_ellipse(ax, mean, cov, scale=1.0, min_size=0, color='0.7'):
    '''Draw covarice ellipses in final Map'''

    eigenvals, eigenvecs = np.linalg.eig(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    angle = math.degrees(math.atan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Compute ellipse axes
    width = max(min_size, 2 * scale * np.sqrt(eigenvals[0]))
    height = max(min_size, 2 * scale * np.sqrt(eigenvals[1]))

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

def get_color_by_uncertainty(uncertainty, min_unc=1, max_unc=100, base='green'):
    # Normalize uncertainty to [0, 1], invert so higher uncertainty is darker
    norm = np.clip((uncertainty - min_unc) / (max_unc - min_unc), 0, 1)
    intensity = 1.0 - norm  # 1=bright, 0=dark
    if base == 'green':
        return (0, intensity, 0)
    elif base == 'blue':
        return (0, 0, intensity)
    else:
        return (intensity, intensity, intensity)

# -----------------------------------------------------------------------------------------------------------------
# ROS

def read_bag_data(bag_file):
    import rosbag

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
        angular_z = msg.twist.twist.angular.z
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
    #return obs_times, obs_data '''Use this line only for camera stochastics'

def read_bag_obs_data(bag_file):
    import rosbag

    bag = rosbag.Bag(bag_file)
    obs_times = []
    obs_data = []

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
    if obs_times:
        time_bias = obs_times[0]
        obs_times = [t - time_bias for t in obs_times]
        for entry in obs_data:
            entry[0] -= time_bias

    return obs_times, obs_data

def select_bag_file_from_list():
    import tkinter as tk
    from tkinter import ttk

    bag_options = [
        'L-30-05-2025.bag',
        'square2-30-05-2025.bag',
        'straight.bag',
        'straight731.bag',
        'L.bag',
        'Lreturn.bag',
        'map1.bag',
        'map2.bag'
    ]
    selected_bag = {'value': None}

    def on_select():
        selected_bag['value'] = combo.get()
        root.destroy()

    root = tk.Tk()
    root.geometry('+900+500')  # (x=400, y=200) - change these values as needed
    root.title('Select ROS Bag File')
    tk.Label(root, text='Choose a bag file:').pack(padx=10, pady=10)
    combo = ttk.Combobox(root, values=bag_options, state='readonly')
    combo.pack(padx=10, pady=5)
    combo.current(0)
    tk.Button(root, text='OK', command=on_select).pack(pady=10)
    root.mainloop()
    return selected_bag['value']

def set_data_conditions_from_bag(bag_name):
    '''
    Sets dataset1, straight_trajectory, L_trajectory, square_trajectory, inverse, just_mapping
    based on the bag file name.
    Returns a tuple of these flags.
    '''
    
    dataset1 = False
    straight_trajectory = False
    L_trajectory = False
    square_trajectory = False
    inverse = False
    just_mapping = False
    camera_offset = 0

    # Dataset 1
    if '30-05-2025' in bag_name:
        dataset1 = True
        if 'L-' in bag_name:
            L_trajectory = True
        elif 'square' in bag_name:
            square_trajectory = True
            camera_offset = -math.pi/2
    # Dataset 2
    elif 'straight' in bag_name:
        dataset1 = False
        straight_trajectory = True
        if '731' in bag_name:
            inverse = True

    elif 'L.bag' in bag_name:
        dataset1 = False
        L_trajectory = True
        if 'return' in bag_name:
            inverse = True

    elif 'map1' in bag_name or 'map2' in bag_name:
        dataset1 = False
        just_mapping = True
    else:
        print('Unknown bag type, please set data conditions manually if needed.')
    return dataset1, straight_trajectory, L_trajectory, square_trajectory, inverse, just_mapping, camera_offset


# -----------------------------------------------------------------------------------------------------------------
# Test

def test_transform_landmarks():
    import numpy as np
    import matplotlib.pyplot as plt

    # Create real landmarks (Nx2)
    real = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

    # Apply known rotation and translation to create estimated landmarks
    theta = np.radians(30)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    t = np.array([2, 3])
    estimated = (real @ R.T) + t

    # Use your function to align estimated to real
    estimated_aligned = transform_landmarks(estimated, real)

    # Plot for visual check
    plt.figure()
    plt.scatter(real[:, 0], real[:, 1], c='green', label='Real')
    plt.scatter(estimated[:, 0], estimated[:, 1], c='red', marker='x', label='Estimated (before)')
    plt.scatter(estimated_aligned[:, 0], estimated_aligned[:, 1], c='blue', marker='o', label='Estimated (aligned)')
    plt.legend()
    plt.title('Test transform_landmarks')
    plt.axis('equal')
    plt.show()

    # Print for numeric check
    print('Real:\n', real)
    print('Estimated (before):\n', estimated)
    print('Estimated (aligned):\n', estimated_aligned)
    print('Alignment error (should be small):', np.linalg.norm(estimated_aligned - real))

# Call this function to test
if __name__ == '__main__':
    test_transform_landmarks()



