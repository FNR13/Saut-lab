import math
import numpy as np
import matplotlib.pyplot as plt

import pygame


from utils import wrap_angle_rad
from utils import pose_estimation
from utils import compute_weight
from utils import update_particles
from utils import resample
from utils import compute_displazament
from utils import compute_angle_ratio
from utils import compute_uncertainty
from utils import update_uncertainties
from utils import resample_uncertainties


from robot import Robot
from carSensor import CarSensor
from env import Envo
from landmarks import Landmarks

from ekf import update_landmark, draw_covariance_ellipse

def main():

    pygame.init()

    # Inizialization
    n_particles = 100
    particles = np.zeros((n_particles, 1, 3), dtype=float) 
    '''
    First column is the particle
    Second column is the time instant
    Third column are the three components of the state (x,y,theta)
    All of them start in the pose (0,0,0)
    '''
    uncertainties  = np.zeros((n_particles,1,3,3),dtype = float)
    '''
    First column is the particle
    Second column is the time instant
    Then, each cell represents the uncertanty matrix of a particle in a concrete instant of time
    As the initial pose is (0,0,0), our uncertity there will be 0. 
    '''
    weights = np.ones(n_particles, dtype=float)/n_particles
    most_probable = 0
    dim = (1200, 800)
    env = Envo(dim)
    start = (400, 200)
    rob = Robot(start, "Robot.png", 0.01 * 3779.52)
    previous_theta = rob.theta
    previous_x = rob.x
    previous_y = rob.y
    landmarks = Landmarks(num_landmarks=10, window_size=(dim[0], dim[1]))

    sensor = CarSensor(
        car_width=rob.wd,  # your robot width
        sensor_width=200,  # sensor trapezoid width in pixels, tweak as needed
        sensor_reach=150,  # sensor reach in pixels, tweak as needed
        color=(0, 255, 0)  # green trapezoid
    )

    clock = pygame.time.Clock()
    highlighted = set()

    dt = 0
    prevtime = pygame.time.get_ticks()

    run = True

    #DELETE
    observation = np.ones(2, dtype=float) #this must be read from the sensor:
                                        #the first value must be the distance between the sensor and the landmark
                                        #the second value must be the angle  
    variance = np.ones(2, dtype=float) #this is the varince in the sensor: first regarding distance and second angle (EKF unit)

    time_instant = 1 
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        rob.update_velocities(keys)
        rob.update_kinematics(dt)

        dt = clock.tick(60) / 1000.0
        prevtime = pygame.time.get_ticks()

        env.win.fill(env.white)
        landmarks.draw(env.win)
        rob.draw(env.win)
        env.trajectory((rob.x, rob.y))
        env.write(int(rob.velL), int(rob.velR), rob.theta)

        sensor.draw(env.win, (rob.x, rob.y), rob.theta)

        visible_landmarks = sensor.filter_landmarks(landmarks.get_positions(), (rob.x, rob.y), rob.theta)
        Q_cov = np.diag([20.0, np.radians(30)])

        #Path estimation
        new_pose = np.zeros((n_particles, 1, 3), dtype=float) 
        new_uncertainty = np.zeros((n_particles, 1, 3, 3), dtype=float)

        for k in range(n_particles): #For each particle

            #Delta D
            delta_D = compute_displazament(rob.x,rob.y,previous_x,previous_y)

            # Delta theta 
            delta_theta = compute_angle_ratio(rob.theta,previous_theta) #CHECK if it is in rad!
            
            # Pose estimation
            new_pose[k,0,:] = pose_estimation(delta_D,delta_theta,particles[k,time_instant-1,:])

            # Uncertainty of the estimaton
            new_uncertainty[k,0,:,:] = compute_uncertainty(uncertainties[k,time_instant-1,:,:].copy(),delta_D,previous_theta)

            for idx, pos in enumerate(landmarks.get_positions()):
                # Draw true landmark position as red circle for ground truth
                pygame.draw.circle(env.win, (255, 0, 0), (int(pos[0]), int(pos[1])), 4)

                if pos in visible_landmarks:
                    dx = pos[0] - rob.x
                    dy = pos[1] - rob.y
                    rng = math.hypot(dx, dy)
                    brg = wrap_angle_rad(math.atan2(dy, dx) - rob.theta)
                    z = np.array([rng, brg, idx])

                    if np.isnan(rob.lm[idx, 0]):
                        lx = rob.x + rng * math.cos(brg + rob.theta)
                        ly = rob.y + rng * math.sin(brg + rob.theta)
                        rob.lm[idx, :] = [lx, ly]
                        rob.lmP[2 * idx:2 * idx + 2, :] = np.eye(2) * 100.0
                        rob.lm_observation_count[idx] = 1  # Initialize count
                    else:
                        rob = update_landmark(rob, z, Q_cov)
                        rob.lm_observation_count[idx] += 1

            for i in range(len(rob.lm)):
                if not np.isnan(rob.lm[i, 0]):
                    mean = rob.lm[i, :]
                    cov = rob.lmP[2 * i:2 * i + 2, :]

                    # Compute ellipse confidence for color
                    uncertainty = np.mean(np.diag(cov[:2, :2]))
                    if uncertainty < 30:
                        color = (0, 255, 0)  # Green = confident
                    elif uncertainty < 80:
                        color = (255, 165, 0)  # Orange = medium
                    else:
                        color = (255, 0, 0)  # Red = uncertain

                    draw_covariance_ellipse(env.win, mean, cov, color=color)

                    # Optional: show uncertainty as text
                    font = pygame.font.SysFont(None, 16)
                    txt = font.render(f"{uncertainty:.1f}", True, (0, 0, 0))
                    env.win.blit(txt, (mean[0] + 5, mean[1] - 5))

                    # Console log (optional)
                    print(f"Landmark {i}: Obs={rob.lm_observation_count[i]} | Cov={np.diag(cov[:2, :2])}")
                    
                    weights[k] = compute_weight(mean,observation,new_pose[k,0,:],variance)
                    weight_previous = weights[k]

            # Weight computing
            # TO DO: add the obsevation taken from the sensor 
            #1. Know which landmark we're looking at (index)
            #2. Get the current distance and the angle to that landmark 
            #3. Compare with the previous measurement
            #4. If we're lookin at a landmark, we update the weight, otherwhise, the weight remains 

            #weights[k] = compute_weight(observation,new_pose[k,0,:],variance)

        # Particle updating
        particles = update_particles(particles, new_pose)

        # Uncertainties updating
        uncertainties = update_uncertainties(uncertainties, new_uncertainty)

        # Weights normalization
        weights = weights/np.sum(weights)

        # Resampling
        particles = resample(particles,weights,n_particles)
        uncertainties = resample_uncertainties(uncertainties,weights,n_particles)
        
        most_probable = np.argmax(weights)
        time_instant += 1
        previous_theta = rob.theta
        previous_x = rob.x
        previous_y = rob.y
                
        pygame.display.update()

    pygame.quit()

    # Plot most probable path
    print('\nMost probable particle:',most_probable)
    print('Wei¡gth of the most probable particle',weights[most_probable]/np.sum(weights))
    plt.plot(particles[most_probable,:,0],particles[most_probable,:,1],label='Mean path')
    plt.xlabel('X')
    plt.ylabel('Y') 
    plt.show()
    print('\n')
    print(uncertainties[most_probable,-1,:,:])

    for t in range(uncertainties.shape[1]):
        if t % 100 == 0:
            mean = np.array([particles[most_probable,t,0],particles[most_probable,t,1]])
            cov = np.diag([uncertainties[most_probable,t,0,0], uncertainties[most_probable,t,1,1]])
            draw_covariance_ellipse(env.win, mean, cov, color=(255, 0, 0))
    
# end main



if __name__ == "__main__":
    main()
