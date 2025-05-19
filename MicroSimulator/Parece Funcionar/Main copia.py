import math
import numpy as np
import matplotlib.pyplot as plt

import pygame


from utils import wrap_angle_rad

from robot import Robot, Particle
from carSensor import CarSensor
from env import Envo
from landmarks import Landmarks

from ekf import update_landmark, draw_covariance_ellipse, compute_jacobians

def compute_weight(particle, z, Q_cov):
    lm_id = int(z[2])
    xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
    Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

    zp, _, Hf, Sf = compute_jacobians(particle, xf, Pf, Q_cov)

    dx = z[0:2].reshape(2, 1) - zp
    dx[1, 0] = wrap_angle_rad(dx[1, 0])

    try:
        inv_Sf = np.linalg.inv(Sf)
    except np.linalg.LinAlgError:
        return 1e-8  # Small weight if matrix is singular

    num = np.exp(-0.5 * dx.T @ inv_Sf @ dx)[0, 0]
    den = 2.0 * math.pi * np.sqrt(np.linalg.det(Sf))

    weight = num / den
    return max(weight, 1e-8)  # Avoid zero weights

def update_with_observation(particles, z_all, Q_cov, N_PARTICLE):
    for iz in range(z_all.shape[1]):  # each column is one measurement
        z = z_all[:, iz]
        lm_id = int(z[2])

        for ip in range(N_PARTICLE):
            particle = particles[ip]

            if np.isnan(particle.lm[lm_id, 0]):
                # Initialize landmark
                rng, brg = z[0], z[1]
                lx = particle.x + rng * math.cos(brg + particle.theta)
                ly = particle.y + rng * math.sin(brg + particle.theta)
                particle.lm[lm_id, :] = [lx, ly]
                particle.lmP[2 * lm_id:2 * lm_id + 2, :] = np.eye(2) * 100.0
                particle.lm_observation_count[lm_id] = 1
            else:
                # Known landmark: update weight and landmark
                w = compute_weight(particle, z, Q_cov)
                particle.weight *= w
                particle = update_landmark(particle, z, Q_cov)

            particles[ip] = particle

    return particles

def normalize_weight(particles):
    total_weight = sum(p.weight for p in particles)
    if total_weight == 0:
        total_weight = 1e-8
    for p in particles:
        p.weight /= total_weight
    return particles

def calc_final_state(particles):
    x_est = np.zeros((3, 1))  # [x, y, theta]

    particles = normalize_weight(particles)

    for p in particles:
        x_est[0, 0] += p.weight * p.x
        x_est[1, 0] += p.weight * p.y
        x_est[2, 0] += p.weight * p.theta

    x_est[2, 0] = wrap_angle_rad(x_est[2, 0])

    return x_est

def resampling(particles):
    particles = normalize_weight(particles)
    N = len(particles)
    weights = np.array([p.weight for p in particles])

    r = np.random.uniform(0, 1.0 / N)
    c = weights[0]
    i = 0
    new_particles = []

    for m in range(N):
        U = r + m * (1.0 / N)
        while U > c:
            i += 1
            c += weights[i]
        # Copy particle
        p = particles[i]
        new_p = Particle(p.x, p.y, p.theta, weight=1.0, robot_width=p.wd)
        new_p.lm = np.copy(p.lm)
        new_p.lmP = np.copy(p.lmP)
        new_p.lm_observation_count = np.copy(p.lm_observation_count)
        new_particles.append(new_p)

    return new_particles

def main():
    pygame.init()
    dim = (1200, 800)
    env = Envo(dim)
    start = (400, 200)
    rob = Robot(start, "Robot.png", 0.01 * 3779.52)
    landmarks = Landmarks(num_landmarks=10, window_size=(dim[0], dim[1]))

    sensor = CarSensor(
        car_width=rob.wd,
        sensor_width=200,
        sensor_reach=150,
        color=(0, 255, 0)
    )

    N_PARTICLES = 1000
    particles = [
        Particle(rob.x, rob.y, rob.theta, robot_width=rob.wd)
        for _ in range(N_PARTICLES)
    ]

    clock = pygame.time.Clock()
    dt = 0

    Q_cov = np.diag([20.0, np.radians(30)])  # Measurement noise

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        rob.update_velocities(keys)
        rob.update_kinematics(dt)

        # --- Motion update for each particle
        for p in particles:
            p.motion_update(rob.velL, rob.velR, dt, noise_std=(1.0, 1.0, 0.02))

        dt = clock.tick(60) / 1000.0  # delta time in seconds

        env.win.fill(env.white)
        landmarks.draw(env.win)
        rob.draw(env.win)
        for p in particles:
            p.draw(env.win, color=(0, 0, 255))  # Blue particles

        env.trajectory((rob.x, rob.y))
        env.write(int(rob.velL), int(rob.velR), rob.theta)

        sensor.draw(env.win, (rob.x, rob.y), rob.theta)

        # --- Landmark detection & particle observation update
        visible_landmarks = sensor.filter_landmarks(landmarks.get_positions(), (rob.x, rob.y), rob.theta)

        # Gather observations: z = [range, bearing, landmark_id]
        z_all = []
        for idx, pos in enumerate(landmarks.get_positions()):
            pygame.draw.circle(env.win, (255, 0, 0), (int(pos[0]), int(pos[1])), 4)  # ground truth

            if pos in visible_landmarks:
                dx = pos[0] - p.x
                dy = pos[1] - p.y
                rng = math.hypot(dx, dy) + 0 * np.random.normal(0, np.sqrt(Q_cov[0, 0]))
                brg = wrap_angle_rad(math.atan2(dy, dx) - p.theta + np.random.normal(0, np.sqrt(Q_cov[1, 1])))
                z_all.append([rng, brg, idx])

        if z_all:
            z_all = np.array(z_all).T  # shape (3, num_obs)
            particles = update_with_observation(particles, z_all, Q_cov, N_PARTICLES)

            # Normalize weights before resampling
            particles = normalize_weight(particles)

            # Resample
            particles = resampling(particles)

            # Normalize again (resampling resets to weight 1)
            particles = normalize_weight(particles)


        selected_particle = max(particles, key=lambda p: p.weight)
        for p in particles:
            if p is not selected_particle:
                p.draw(env.win, color=(150, 150, 255))  # Light blue

        selected_particle.draw(env.win, color=(255, 0, 255))  # Magenta

        for i in range(len(selected_particle.lm)):
            if not np.isnan(selected_particle.lm[i, 0]):
                mean = selected_particle.lm[i, :]
                cov = selected_particle.lmP[2 * i:2 * i + 2, :]

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
                env.win.blit(txt, (mean[0] + 5, mean[1] - 5))

                print(f"Landmark {i}: Obs={selected_particle.lm_observation_count[i]} | Cov={np.diag(cov[:2, :2])}")

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()

