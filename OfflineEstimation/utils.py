import math
import numpy as np

import rosbag


def wrap_angle_rad(angle):
    """Wrap angle to [0, 2π)"""
    return angle % (2 * math.pi)


def update_paths(paths, new_pose):
    '''Add the last pose of each particle to its path'''
    return np.concatenate((paths, new_pose), axis=1)

def resample_paths(paths,indexes):
    '''Resample the particles with replacement'''
    # Resampling 
    paths = paths[indexes, :, :]
    return paths

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