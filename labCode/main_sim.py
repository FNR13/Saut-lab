import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from matplotlib.patches import Ellipse

import pygame

from classUtils.utils import wrap_angle_rad, draw_fastslam_particles, draw_covariance_ellipse, update_paths, resample_paths, draw_ellipse

from classSimulation.robot import Robot
from classSimulation.carSensor import CarSensor
from classSimulation.env import Envo
from classSimulation.landmarks import Landmarks

from classAlgorithm.fastslam import FastSLAM

def main():

    # robot_initial_pose = (200, 200, -math.pi/2)
    robot_initial_pose = (0, 0, 0)
    gt_path = []  


    # FastSLAM initialization
    N_PARTICLES = 100
    particles_odometry_uncertainty = (0.001, 0.05)  # (speed, anngular rate)
    landmarks_initial_uncertainty = 10  # Initial uncertainty for landmarks
    Q_cov = np.diag([20.0, np.radians(30)]) # Measurement noise for fast slam - for range and bearing
    
    fastslam = FastSLAM(
        robot_initial_pose,
        N_PARTICLES,
        particles_odometry_uncertainty,
        landmarks_initial_uncertainty,
        Q_cov,
    )

    # Array for saving all paths
    paths = np.zeros((N_PARTICLES,1,3),dtype=float)
    
    # Noise implementations
    use_camera_noise = False
    distance_noise_power = 20
    bearing_noise_power = np.radians(5)

    use_odometry_noise = False
    odemetry_noise_power = 0.001

    # Simulation Initiations
    pygame.init()
    dim = (1200, 800)
    env = Envo(dim)

    rob = Robot((robot_initial_pose[0], robot_initial_pose[1], robot_initial_pose[2]), "media/Robot.png", 0.01 * 3779.52)

    sensor = CarSensor(
        car_width=rob.wd,
        sensor_width=200,
        sensor_reach=150,
        color=(0, 255, 0)
    )

    landmarks = Landmarks(num_landmarks=10, window_size=(dim[0], dim[1]))

    real_positions = landmarks.get_positions()

    B = 0
    best_path = 0
    landmarks_uncertainty = 0

    clock = pygame.time.Clock()
    dt = 0

    # -----------------------------------------------------------------------------------------------------------------------------
    # Loop
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # --- Simulation step
        keys = pygame.key.get_pressed()
        rob.update_velocities(keys)
        rob.update_kinematics(dt)

        dt = clock.tick(60) / 1000.0  # delta time in seconds
        gt_path.append((rob.x, rob.y))

        # Draw Simulation objects
        env.win.fill(env.white)
        landmarks.draw(env.win)
        
        rob.draw(env.win)
        sensor.draw(env.win, (rob.x, rob.y), rob.theta)
        env.trajectory((rob.x, rob.y))
        env.write(int(rob.velL), int(rob.velR), rob.theta)

        draw_fastslam_particles(fastslam.particles, env.win)

        # --- Landmark detection
        visible_landmarks = sensor.filter_landmarks(landmarks.get_positions(), (rob.x, rob.y), rob.theta)

        # Gather observations: z = [landmark_id, distance, bearing]
        z_all = []
        for idx, pos in enumerate(landmarks.get_positions()):
            pygame.draw.circle(env.win, (255, 0, 0), (int(pos[0]), int(pos[1])), 4)  # ground truth

            if pos in visible_landmarks:
                
                marker_id = idx

                # Real system camera feed
                dx = pos[0] - rob.x
                dy = pos[1] - rob.y

                distance = math.hypot(dx, dy) #it was range in previous versions but range is an internal functionfro python 
                bearing = wrap_angle_rad(math.atan2(dy, dx) + rob.theta)

                # Add noise
                if use_camera_noise:
                    distance += np.random.normal(0, distance_noise_power) 
                    bearing += np.random.normal(0, bearing_noise_power) 

                z_all.append([marker_id, distance, bearing])
        
        # Add noise to the odometry
        if use_odometry_noise:
            rob.velL += np.random.normal(0, odemetry_noise_power)
            rob.velR += np.random.normal(0, odemetry_noise_power)

        velocity = (rob.velL + rob.velR) / 2.0
        omega = ((rob.velR - rob.velL)/ rob.wd)*-1

        # ----------
        # FastSLAM step
        # ----------
        # Motion update 
        fastslam.predict_particles(velocity, omega, dt)

        new_pose = np.zeros((N_PARTICLES,1,3),dtype=float)
        for i, particle in enumerate(fastslam.particles):
            particle_pose = np.array([particle.x,particle.y,particle.theta])
            new_pose[i,0,:] = particle_pose
        
        # Add the new pose to the path
        paths = update_paths(paths, new_pose)

        # --- Observation update
        if z_all:   
            fastslam.observation_update(z_all)
            fastslam.resampling()

            # Resample the paths in the same manner as the particles
            paths = resample_paths(paths,fastslam.resampled_indexes)
        # ----------

        # Draw the best particle
        selected_particle = fastslam.get_best_particle()
        best_path = fastslam.particles.index(selected_particle)

        draw_fastslam_particles([selected_particle], env.win, color=(255, 0, 255))  # Magenta'''

        if selected_particle.landmarks_id:
            
            # Set we want totransform (landmarks estimated position)
            B = selected_particle.landmarks_position 
            B = np.array([b.flatten() for b in B])
            landmarks_uncertainty = selected_particle.landmarks_position_covariance 

            # Set we want to convert to (real landmarks position)
            A = [] #Real Positions

            for id in selected_particle.landmarks_id:
                A.append(real_positions[id])    

            A = np.array(A)
            
            A_mean = A.mean(axis=0)
            B_mean = B.mean(axis=0)
            A_centered = A - A_mean
            B_centered = B - B_mean

            # Find best rotation
            R, _ = orthogonal_procrustes(B_centered, A_centered)

            # Apply rotation to B
            B_rotated = B_centered @ R

            # Translate to the A coordenates origin 
            B_aligned = B_rotated + A_mean
            
            for marker_id in selected_particle.landmarks_id:

                # get index of the landmark from the list
                idx = selected_particle.landmarks_id.index(marker_id)

                mean = selected_particle.landmarks_position[idx]
                cov = selected_particle.landmarks_position_covariance[idx]

                mean_flat = mean.flatten()  
                aux = np.where((B == mean_flat).all(axis=1))[0]
                mean_2 = B_aligned[aux,:]
                mean_2 = mean_2.reshape(-1)

                uncertainty = np.mean(np.diag(cov[:2, :2]))

                if uncertainty < 30:
                    color = (0, 255, 0)
                elif uncertainty < 80:
                    color = (255, 165, 0)
                else:
                    color = (255, 0, 0)
                
                draw_covariance_ellipse(env.win, mean, cov, color=color)
                draw_covariance_ellipse(env.win, mean_2, cov, color=(0,0,0))

                font = pygame.font.SysFont(None, 16)
                txt = font.render(f"{uncertainty:.1f}", True, (0, 0, 0))
                env.win.blit(txt, (mean[0][0] + 5, mean[1][0] - 5))
                env.win.blit(txt, (mean_2[0] + 5, mean_2[1] - 5))

                # print(f"Landmark {selected_particle.landmarks_id[idx]}: Obs={selected_particle.landmarks_observation_count[idx]} | Cov={np.diag([cov[0][0], cov[1][1]])} | Pos={mean[0][0]:.1f}, {mean[1][0]:.1f} | Uncertainty={uncertainty:.1f}")
        
                
        pygame.display.update()
    pygame.quit()
    
    if isinstance(B, np.ndarray):
        fig, ax = plt.subplots()
        
        # Most probable FastSLAM trajectory
        plt.plot(paths[best_path,:,0], -paths[best_path,:,1], label='Most Probable Path')
        
        # Ground truth robot path
        gt_path = np.array(gt_path)
        plt.plot(gt_path[:, 0], -gt_path[:, 1], 'r--', label='Ground Truth Path')
    
        # Landmarks and uncertainty ellipses
        for i in range(len(landmarks_uncertainty)):
            ellipse = draw_ellipse(ax, B[i,:], landmarks_uncertainty[i])
            if i == 0:
                ellipse.set_label('Estimated Landmarks')
    
        plt.title('FastSLAM')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='best')
        plt.grid()
        plt.axis('equal')
        plt.show()
    else:
        print('No landmarks seen')

   

if __name__ == "__main__":
    main()