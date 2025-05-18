import math
import numpy as np

def wrap_angle_rad(angle):
    """Wrap angle to [0, 2π)"""
    return angle % (2 * math.pi)


def pose_estimation(delta_D,delta_theta,previous_pose):
    '''Dynamics fucntion'''
    # TO DO: add the uncertainty 
    
    # Movement noise following a Normal distribution
    mean = np.zeros(3) #zero mean
    cov = np.eye(3)    #identity matrix 
    epsilon = np.random.multivariate_normal(mean, cov) #MAYBE the scale of this estimation is too large for the angle

    # Prediction of the  new pose
    new_pose = np.zeros(3,dtype=float)

    # Orientation
    theta_t = previous_pose[2]
    theta_t_1 = theta_t + delta_theta #+ epsilon[2]
    new_pose[2] = theta_t_1

    # X coordinate 
    x_t = previous_pose[0]
    x_t_1 = x_t + delta_D*np.cos(theta_t) #+ epsilon[0]
    new_pose[0] = x_t_1 

    # Y cordinate
    y_t = previous_pose[1]
    y_t_1 = y_t + delta_D*np.sin(theta_t) #+ epsilon[1]
    new_pose[1] = y_t_1 

    return new_pose


def observation_prediction(pose):
    '''Obserbavility function''' 

    #TO DO: p_x and p_y from the real map, identify the landmark and delete the following line when it's done
    p_x = 3.0 #x coordenate mean of landmark i in a concrete particle 
    p_y = 2.0 #y coordenate mean of landmark i in a concrete particle

    # Measurement noise following a Normal distribution
    mean = np.zeros(2) #zero mean
    cov = np.eye(2)    #identity matrix 
    epsilon = np.random.multivariate_normal(mean, cov)
    
    z_hat = np.zeros(2,dtype = float)

    # Range
    r_hat = math.sqrt(((p_x - pose[0])**2) + ((p_y - pose[1])**2)) #+ epsilon[0]
    z_hat[0] = r_hat
    
    # Angle
    phi_hat = math.atan((p_y - pose[1])/(p_x - pose[0])) - pose[2] #+ epsilon[1]
    z_hat[1] = phi_hat

    return z_hat


def compute_weight(observation, pose, variance):

    # TO DO: discover the variance of the sensor: are they const or dynamic?
    '''
    Input:
     - real measurements taken from the sensor
     - the variance of the sensor: both in distance ([0]) and angle ([1])
     - the estimated pose

    Output:
     - weght = probability an observaion given the robot's pose
    '''

    # Predictions based on the estimated position of the robot 
    z_predicted = observation_prediction(pose)
    r_hat = z_predicted[0] #distance between sensor and landmark 
    phi_hat = z_predicted[1] #angle between sensor and landmark 

    # Real position based on the sensor data
    r = observation[0]
    phi = observation[1]

    # Compute the probabilities for each variable (Gaussian distribution) 
    p_zr_given_xi = (1/math.sqrt(2*math.pi*variance[0]))*math.exp(-(r-r_hat)/2*variance[0]) 
    p_zphi_given_xi = (1/math.sqrt(2*math.pi*variance[1]))*math.exp(-(phi-phi_hat)/2*variance[1]) 

    # Compute joined probabilities of incorrelated variables
    return p_zr_given_xi*p_zphi_given_xi


def update_particles(particles, new_pose):
    '''Add the last poses of each particle to its particle'''
    return np.hstack((particles, new_pose))


def resample(particles,weights,n_particles):
    '''Resample the particles with replazament'''
    
    # Weights normalization
    weights = weights/np.sum(weights)

    # Resampling 
    N = n_particles
    indexes = np.random.choice(N, size=N, p=weights) 
    particles = particles[indexes, :, :]
    return particles




