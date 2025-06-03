import math
import numpy as np
import matplotlib.pyplot as plt

import pygame

from classUtils.utils import *

from classSimulation.robot import Robot
from classSimulation.carSensor import CarSensor
from classSimulation.env import Envo
from classSimulation.landmarks import Landmarks

from classAlgorithm.fastslam import FastSLAM
from classAlgorithmRangeOnly.fastslam import FastSLAM_RO

# -----------------------------------------------------------------------------------------------------------------
def main():

    robot_initial_pose = (200, 300, math.pi/2)
    # robot_initial_pose = (0, 0, 0)

    use_range_only_fastslam = True

    # FastSLAM initialization
    N_PARTICLES = 100
    particles_odometry_uncertainty = (0.001, 0.05)  # (speed, anngular rate)
    landmarks_initial_uncertainty = 100  # Initial uncertainty for landmarks
    Q_cov = np.diag([20.0, np.radians(30)]) # Measurement noise for fast slam - for range and bearing

    if use_range_only_fastslam:
        print("Using Range Only FastSLAM")
         # Range Only FastSLAM parameters    
        sensor_fov = 60 #vision range of the camera in º
        Q_cov = 20.0 # Measurement noise for fast slam - for range

        # Range Only FastSLAM initialization
        fastslam = FastSLAM_RO(
            robot_initial_pose,
            N_PARTICLES,
            particles_odometry_uncertainty,
            landmarks_initial_uncertainty,
            Q_cov, 
            sensor_fov,
        )
    else:
        print("Using FastSLAM")
        fastslam = FastSLAM(
            robot_initial_pose,
            N_PARTICLES,
            particles_odometry_uncertainty,
            landmarks_initial_uncertainty,
            Q_cov,
        )

        # Noise implementations
    use_camera_noise = False
    distance_noise_power = 20
    bearing_noise_power = np.radians(5)

    use_odometry_noise = True
    odemetry_noise_power = 0.05

    # Gound truth path
    gt_path = []  

    # Array for saving all paths
    paths = np.zeros((N_PARTICLES,1,3),dtype=float)

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

    clock = pygame.time.Clock()
    dt = 0

    # -----------------------------------------------------------------------------------------------------------------
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

                if use_range_only_fastslam:
                    # For Range Only FastSLAM, we only need the distance
                    z_all.append([marker_id, distance])
                else:
                    z_all.append([marker_id, distance, bearing])
        
        # Add noise to the odometry
        if use_odometry_noise:
            rob.velL += np.random.normal(0, odemetry_noise_power)
            rob.velR += np.random.normal(0, odemetry_noise_power)

        velocity = (rob.velL + rob.velR) / 2.0
        omega = ((rob.velR - rob.velL)/ rob.wd)*-1

        # -----------------------------------------------------------------------------------------------------------------
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
        
        # -----------------------------------------------------------------------------------------------------------------
        # Drawing 

        # Get the best particle
        best_particle = fastslam.get_best_particle()
        best_path = fastslam.particles.index(best_particle)
        draw_fastslam_particles([best_particle], env.win, color=(255, 0, 255))  # Magenta'''

        if best_particle.landmarks_id:
            
            # Set we want totransform (landmarks estimated position)
            # Set we want to convert to (real landmarks position)
            identified_real_landmarks = [] #Real Positions

            for id in best_particle.landmarks_id:
                identified_real_landmarks.append(real_positions[id])    

            identified_real_landmarks = np.array(identified_real_landmarks)
             
            aligned_estimated_landmarks, estimated_center, real_center, Rotation = transform_landmarks(best_particle.landmarks_position, identified_real_landmarks)
            
            for marker_id in best_particle.landmarks_id:

                # get index of the landmark from the list
                idx = best_particle.landmarks_id.index(marker_id)

                mean_estimated = best_particle.landmarks_position[idx]
                cov_estimated = best_particle.landmarks_position_covariance[idx] 

                mean_alligned = aligned_estimated_landmarks[idx,:]
                mean_alligned = mean_alligned.reshape(-1)

                uncertainty = np.mean(np.diag(cov_estimated[:2, :2]))

                if uncertainty < 30:
                    color = (0, 255, 0)
                elif uncertainty < 80:
                    color = (255, 165, 0)
                else:
                    color = (255, 0, 0)
                
                draw_covariance_ellipse(env.win, mean_estimated, cov_estimated, color=color)
                draw_covariance_ellipse(env.win, mean_alligned, cov_estimated, color=(0,178,0))

                font = pygame.font.SysFont(None, 16)
                txt = font.render(f"{uncertainty:.1f}", True, (0, 0, 0))
                env.win.blit(txt, (mean_estimated[0][0] + 5, mean_estimated[1][0] - 5))
                env.win.blit(txt, (mean_alligned[0] + 10, mean_alligned[1] - 10))

                # print(f"Landmark {selected_particle.landmarks_id[idx]}: Obs={selected_particle.landmarks_observation_count[idx]} | Cov={np.diag([cov[0][0], cov[1][1]])} | Pos={mean[0][0]:.1f}, {mean[1][0]:.1f} | Uncertainty={uncertainty:.1f}")
        
                
        pygame.display.update()
    pygame.quit()

    # -----------------------------------------------------------------------------------------------------------------
    # After the simulation, we can plot the results

    ground_truth_color = 'red'
    particle_color = 'black'
    aligned_estimation_color = 'blue'

    if best_particle.landmarks_id:
        fig, ax = plt.subplots()
        
            # Note when plotting it is needed to convert the coordinates from (x, y) to (x, -y)

        # Ground truth
            # Path 
        gt_path = np.array(gt_path)
        plt.plot(gt_path[:, 0], -gt_path[:, 1], 
                 'r--', label='Ground Truth Path', linewidth=2)
            # Landamrks
        ax.scatter(identified_real_landmarks[:,0], -identified_real_landmarks[:,1],
           facecolors='none', edgecolors=ground_truth_color, marker='o', label='Real Landmarks', linewidths=2)
        
        # Estimation
            # Most probable FastSLAM trajectory
        # Extract the most probable path (Nx2)
        most_probable_path = paths[best_path, :, :2]

        # Center the path using the same estimated_center as for landmarks
        centered_path = most_probable_path - estimated_center

        # Apply the same rotation and translation as for landmarks
        # NOTE: IF just one landmark was seen, the rotation will be identity, so it won work
        aligned_most_probable_path = (centered_path @ Rotation) + real_center

        plt.plot(most_probable_path[:,0], -most_probable_path[:,1], 
                 particle_color, label='Estimated Most Probable Path', linewidth=1)
        plt.plot(aligned_most_probable_path[:,0], -aligned_most_probable_path[:,1], 
                 aligned_estimation_color, label='Aligned Most Probable Path', linewidth=1)
        
        # Estimated landmarks
            # Estimated landmarks with elipses
        # for marker_id in best_particle.landmarks_id:

        #         # get index of the landmark from the list
        #         idx = best_particle.landmarks_id.index(marker_id)

        #         mean_estimated = best_particle.landmarks_position[idx]
        #         mean_estimated = [mean_estimated[0][0], -mean_estimated[1][0]] # Transform to (x, -y)

        #         mean_alligned = aligned_estimated_landmarks[idx,:]
        #         mean_alligned = mean_alligned.reshape(-1)
        #         mean_alligned = [mean_alligned[0], -mean_alligned[1]] # Transform to (x, -y)

        #         cov_estimated = best_particle.landmarks_position_covariance[idx] 
        #         uncertainty = np.mean(np.diag(cov_estimated[:2, :2]))

        #         if uncertainty < 30:
        #             color = (0, 1, 0)
        #         elif uncertainty < 80:
        #             color = (1, 165/255, 0)
        #         else:
        #             color = (1, 0, 0)
                
        #         color_est = get_color_by_uncertainty(uncertainty, base=particle_color)
        #         color_aligned = get_color_by_uncertainty(uncertainty, base=aligned_estimation_color)

        #         draw_ellipse(ax, mean_estimated, cov_estimated, color=color_est)
        #         draw_ellipse(ax, mean_alligned, cov_estimated, color=color_aligned)

            # Estimated landmarks points
        estimated = np.array([
            [landmark[0][0], landmark[1][0]] for landmark in best_particle.landmarks_position])
        aligned = np.array([
            [landmark[0], landmark[1]] for landmark in aligned_estimated_landmarks])

        ax.scatter(estimated[:, 0], -estimated[:, 1], marker='x', c=particle_color, label='Estimated Landmarks')
        ax.scatter(aligned[:, 0], -aligned[:, 1], marker='x', c=aligned_estimation_color, label='Aligned Landmarks')

        plt.title('FastSLAM')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='best')
        plt.grid()
        plt.axis('equal')
        plt.show()
    else:
        print('No landmarks seen')

# -----------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    