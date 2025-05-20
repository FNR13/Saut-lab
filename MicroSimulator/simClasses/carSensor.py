import pygame

import math

class CarSensor:
    def __init__(self, car_width, sensor_width, sensor_reach, color=(0, 255, 0)):
        self.car_width = car_width
        self.sensor_width = sensor_width
        self.sensor_reach = sensor_reach
        self.color = color

    def compute_trapezoid(self, x, y, theta):
        # Half widths
        half_car = self.car_width / 2
        half_sensor = self.sensor_width / 2

        # Sensor front points
        front_left = (
            x + self.sensor_reach * math.cos(theta) - half_sensor * math.sin(theta),
            y - self.sensor_reach * math.sin(theta) - half_sensor * math.cos(theta),
        )
        front_right = (
            x + self.sensor_reach * math.cos(theta) + half_sensor * math.sin(theta),
            y - self.sensor_reach * math.sin(theta) + half_sensor * math.cos(theta),
        )

        # Sensor rear points (near robot front)
        rear_left = (
            x - half_car * math.sin(theta),
            y - half_car * math.cos(theta),
        )
        rear_right = (
            x + half_car * math.sin(theta),
            y + half_car * math.cos(theta),
        )

        # Return points in order (rear_left, front_left, front_right, rear_right)
        return [rear_left, front_left, front_right, rear_right]

    def draw(self, win, robot_pos, robot_theta):
        polygon = self.compute_trapezoid(robot_pos[0], robot_pos[1], robot_theta)
        pygame.draw.polygon(
            win, self.color,
            [(int(p[0]), int(p[1])) for p in polygon], 2
        )

# -----------------------------------------------------------------------------------------------------------------------------
    # Check for landmark code
    def point_in_polygon(self, point, polygon):
        x, y = point
        inside = False
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if ((y1 > y) != (y2 > y)) and \
                (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
                inside = not inside
        return inside

    def filter_landmarks(self, landmarks_positions, robot_pos, robot_theta):
        polygon = self.compute_trapezoid(robot_pos[0], robot_pos[1], robot_theta)
        visible = []
        for pos in landmarks_positions:
            if self.point_in_polygon(pos, polygon):
                visible.append(pos)
        return visible

