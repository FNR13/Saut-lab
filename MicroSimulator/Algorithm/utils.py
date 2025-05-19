import math
import numpy as np

def wrap_angle_rad(angle):
    """Wrap angle to [0, 2π)"""
    return angle % (2 * math.pi)


def pose_estimation(delta_D,delta_theta,previous_pose):
    '''Dynamics fucntion'''    
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

    return new_pose.copy()


def compute_Jacobian(delta_D, previous_angle):
        Jacobian = np.zeros((3,3),dtype=float)
        Jacobian[0,0] = 1
        Jacobian[1,1] = 1
        Jacobian[2,2] = 1
        Jacobian[0,2] = -delta_D*math.sin(previous_angle)
        Jacobian[1,2] = delta_D*math.cos(previous_angle)

        return Jacobian.copy()


def compute_uncertainty(previous_un, delta_D, previous_angle):

    # Uncerainty associated to the robot's
    new_uncertainty = np.zeros((3,3),dtype=float)

    F = compute_Jacobian(delta_D,previous_angle)

    # Error characterization
    cov = np.eye(3)    #covariance of the noise

    new_uncertainty = F @ previous_un @ F.T + cov

    return new_uncertainty.copy()
 

def observation_prediction(pose,observation):
    '''Obserbavility function''' 

    #TO DO: p_x and p_y from the real map, identify the landmark and delete the following line when it's done
    p_x = observation[0] #x coordenate mean of landmark i in a concrete particle 
    p_y = observation[1] #y coordenate mean of landmark i in a concrete particle

    # Measurement noise following a Normal distribution
    noise_mean = np.zeros(2) #zero mean
    cov = np.eye(2)    #identity matrix 
    epsilon = np.random.multivariate_normal(noise_mean, cov)
    
    z_hat = np.zeros(2,dtype = float)

    # Range
    r_hat = math.sqrt(((p_x - pose[0])**2) + ((p_y - pose[1])**2)) #+ epsilon[0]
    z_hat[0] = r_hat
    
    # Angle
    phi_hat = math.atan((p_y - pose[1])/(p_x - pose[0])) - pose[2] #+ epsilon[1]
    z_hat[1] = phi_hat

    return z_hat


def compute_weight(observation, previous_obs, pose, variance):

    # TO DO: discover the variance of the sensor: are they const or dynamic?
    '''
    Input:
     - real measurements taken from the sensor
     - the variance of the sensor: both in distance ([0]) and angle ([1])
     - the estimated pose

    Output:
     - weght = probability an observaion given the robot's pose
    '''

    # Predictions based on the estimated position of the robot and the previous estimated landmark position
    z_hat = observation_prediction(pose,previous_obs)
    r_hat = z_hat[0] #distance between sensor and landmark 
    phi_hat = z_hat[1] #angle between sensor and landmark 

    # Predictions based on the estimated position of the robot and the current estimated landmark position
    z_current = observation_prediction(pose,observation)
    r_current = z_current[0] #distance between sensor and landmark 
    phi_current = z_current[1] #angle between sensor and landmark 

    # Compute the probabilities for each variable (Gaussian distribution) 
    p_zr_given_xi = (1/math.sqrt(2*math.pi*variance[0]))*math.exp(-(r_current-r_hat)/2*variance[0]) 
    p_zphi_given_xi = (1/math.sqrt(2*math.pi*variance[1]))*math.exp(-(phi_current-phi_hat)/2*variance[1]) 

    # Compute joined probabilities of incorrelated variables
    return p_zr_given_xi*p_zphi_given_xi


def update_particles(particles, new_pose):
    '''Add the last pose of each particle to its particle'''
    return np.hstack((particles, new_pose))


def update_uncertainties(uncertainies, new_uncertainty):
    '''Add the uncertainty associated to the last pose of each particle'''
    return np.hstack((uncertainies, new_uncertainty))


def resample(particles,weights,n_particles):
    '''Resample the particles with replazament'''

    # Resampling 
    N = n_particles
    indexes = np.random.choice(N, size=N, p=weights) 
    particles = particles[indexes, :, :]
    return particles


def resample_uncertainties(uncertainties,weights,n_particles):
    '''Resample the uncertainties associated to the previous particles with replazament'''
  
    # Resampling 
    N = n_particles
    indexes = np.random.choice(N, size=N, p=weights) 
    uncertainties = uncertainties[indexes, :, :, :]
    return uncertainties
           

    

def compute_displazament(current_x,current_y,previous_x,previous_y):
    '''Delta D calculation'''
    delta_Dx = current_x - previous_x
    delta_Dy = current_y - previous_y
    return math.sqrt((delta_Dx**2)+(delta_Dy**2))


def compute_angle_ratio(current_angle, previous_angle):
    '''Delta theta computation'''

    return (current_angle-previous_angle)


