import math
import numpy as np
import matplotlib.pyplot as plt

import pygame

from utils import wrap_angle_rad, draw_fastslam_particles, draw_covariance_ellipse

from robot import Robot
from carSensor import CarSensor
from env import Envo
from landmarks import Landmarks

from fastslam import FastSLAM

def main():

    robot_initial_pose = (400, 200, 0)

    # FastSLAM initialization
    N_PARTICLES = 100
    particles_odometry_uncertanty = (0.001, 0.05)  # (speed, anngular rate)
    landmarks_initial_uncertanty = 100  # Initial uncertainty for landmarks
    Q_cov = np.diag([20.0, np.radians(30)])  # Measurement noise for fast slam - for range and bearing
    
    fastslam = FastSLAM(
        robot_initial_pose,
        N_PARTICLES,
        particles_odometry_uncertanty,
        landmarks_initial_uncertanty,
        Q_cov,
    )
    
    # Noise implementations
    use_camera_noise = False
    range_noise_power = 20
    bearing_noise_power = np.radians(5)

    use_odometry_noise = False
    odemetry_noise_power = 0.001

    # Simulation Initiations
    pygame.init()
    dim = (1200, 800)
    env = Envo(dim)

    rob = Robot((robot_initial_pose[0], robot_initial_pose[1]), "Robot.png", 0.01 * 3779.52)

    sensor = CarSensor(
        car_width=rob.wd,
        sensor_width=200,
        sensor_reach=150,
        color=(0, 255, 0)
    )

    landmarks = Landmarks(num_landmarks=10, window_size=(dim[0], dim[1]))

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

        # Gather observations: z = [landmark_id, range, bearing]
        z_all = []
        for idx, pos in enumerate(landmarks.get_positions()):
            pygame.draw.circle(env.win, (255, 0, 0), (int(pos[0]), int(pos[1])), 4)  # ground truth

            if pos in visible_landmarks:
                
                marker_id = idx

                # Real system camera feed
                dx = pos[0] - rob.x
                dy = pos[1] - rob.y

                range = math.hypot(dx, dy)
                bearing = wrap_angle_rad(math.atan2(dy, dx) - rob.theta)

                # Add noise
                if use_camera_noise:
                    range =+ np.random.normal(0, range_noise_power) 
                    bearing =+ np.random.normal(0, bearing_noise_power) 

                z_all.append([marker_id, range, bearing])
        
        # Add noise to the odometry
        if use_odometry_noise:
            rob.velL += np.random.normal(0, odemetry_noise_power)
            rob.velR += np.random.normal(0, odemetry_noise_power)

        velocity = (rob.velL + rob.velR) / 2.0
        omega = (rob.velR - rob.velL) / rob.wd

        # ----------
        # FastSLAM step
        # --- Motion update 
        fastslam.predict_particles(velocity, omega, dt)

        # --- Observation update
        if z_all:   
            fastslam.observation_update(z_all)
            fastslam.resampling()
        # ----------

        # Draw the best particle
        selected_particle = fastslam.get_best_particle()
        draw_fastslam_particles([selected_particle], env.win, color=(255, 0, 255))  # Magenta

        if selected_particle.landmarks_id:
            for marker_id in selected_particle.landmarks_id:

                # get index of the landmark from the list
                idx = selected_particle.landmarks_id.index(marker_id)

                mean = selected_particle.landmarks_position[idx]
                cov = selected_particle.landmarks_position_covariance[idx]

                uncertainty = np.mean(np.diag(cov[:2, :2]))

                if uncertainty < 30:
                    color = (0, 255, 0)
                elif uncertainty < 80:
                    color = (255, 165, 0)
                else:
                    color = (255, 0, 0)

                draw_covariance_ellipse(env.win, mean, cov, color=color)

                font = pygame.font.SysFont(None, 16)
                txt = font.render(f"{uncertainty:.1f}", True, (0, 0, 0))
                env.win.blit(txt, (mean[0][0] + 5, mean[1][0] - 5))

                print(f"Landmark {selected_particle.landmarks_id[idx]}: Obs={selected_particle.landmarks_observation_count[idx]} | Cov={np.diag([cov[0][0], cov[1][1]])} | Pos={mean[0][0]:.1f}, {mean[1][0]:.1f} | Uncertainty={uncertainty:.1f}")
        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()