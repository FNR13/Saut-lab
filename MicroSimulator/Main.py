import math
import pygame
import random


def wrap_angle_rad(angle):
    """Wrap angle to [0, 2π)"""
    return angle % (2 * math.pi)


class Envo:
    def __init__(self,dimensions):
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.green = (0,255,0)
        self.blue = (0,0,255)
        self.red = (255,0,0)
        self.yellow = (255,255,0)

        self.length = dimensions[0]
        self.width = dimensions[1]

        pygame.display.set_caption("Differential Drive Robot")
        self.win = pygame.display.set_mode((self.length,self.width))

        self.font = pygame.font.Font('freesansbold.ttf',35)
        self.text = self.font.render('default',True,self.white,self.black)
        self.textrect = self.text.get_rect()
        self.textrect.center=(dimensions[1]-175, dimensions[0]-500)
        self.traj_set=[]

    def write (self, velL, velR, theta):
        txt = f"Vel L = {velL/3779.52:.3f} Vel R ={velR/3779.52:.3f} theta = {int(math.degrees(wrap_angle_rad(theta)))}"
        self.text=self.font.render(txt,True,self.white,self.black)
        self.win.blit(self.text,self.textrect)

    def trajectory(self,pos):
        for i in range(0,len(self.traj_set)-1):
            pygame.draw.line(self.win,self.red,(self.traj_set[i][0],self.traj_set[i][1]),(self.traj_set[i+1][0],self.traj_set[i+1][1]))
        if self.traj_set.__sizeof__()>300000:
            self.traj_set.pop(0)
        self.traj_set.append(pos)


class Robot:
    def __init__(self, startpoint, robotimg, width):
        self.met2pix = 3779.52
        self.wd = width
        self.x = startpoint[0]
        self.y = startpoint[1]
        self.theta = 0
        self.velL = 0.01 * self.met2pix
        self.velR = 0.01 * self.met2pix
        self.max = 0.05 * self.met2pix
        self.min = 0.002 * self.met2pix

        self.imge = pygame.image.load(robotimg).convert_alpha()
        self.rotate = self.imge
        self.rect = self.rotate.get_rect(center=(self.x, self.y))

    def draw(self,win):
        win.blit(self.rotate,self.rect)

    def update_velocities(self, keys):
        increment = 0.0001 * self.met2pix

        if keys[pygame.K_KP4]:  # Increase left
            self.velL = min(self.velL + increment, self.max)
        if keys[pygame.K_KP1]:  # Decrease left
            self.velL = max(self.velL - increment, self.min)
        if keys[pygame.K_KP6]:  # Increase right
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_KP3]:  # Decrease right
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_KP5]:  # Increase both
            self.velL = min(self.velL + increment, self.max)
            self.velR = min(self.velR + increment, self.max)
        if keys[pygame.K_KP2]:  # Decrease both
            self.velL = max(self.velL - increment, self.min)
            self.velR = max(self.velR - increment, self.min)
        if keys[pygame.K_SPACE]:  # Equalize both velocities
            avg = (self.velL + self.velR) / 2
            self.velL = self.velR = avg

    def update_kinematics(self):
        self.x += ((self.velL + self.velR) / 2) * math.cos(self.theta) * dt
        self.y -= ((self.velL + self.velR) / 2) * math.sin(self.theta) * dt
        self.theta += (self.velR - self.velL) / self.wd * dt
    
        self.rotate = pygame.transform.rotozoom(self.imge, math.degrees(self.theta), 1)
        self.rect = self.rotate.get_rect(center=(self.x, self.y))
        

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

    def draw(self, win, robot_pos, robot_theta):
        polygon = self.compute_trapezoid(robot_pos[0], robot_pos[1], robot_theta)
        pygame.draw.polygon(
            win, self.color,
            [(int(p[0]), int(p[1])) for p in polygon], 2
        )

        
class Landmarks:
    def __init__(self, num_landmarks, window_size, radius=8, color=(0, 0, 255), min_dist=100):
        """
        num_landmarks: how many landmarks to generate
        window_size: tuple (width, height) for placing landmarks inside screen
        radius: radius of each landmark circle
        color: RGB color tuple for landmarks
        min_dist: minimum allowed distance between landmarks
        """
        self.num = num_landmarks
        self.width, self.height = window_size
        self.radius = radius
        self.color = color
        self.min_dist = min_dist
        
        self.positions = []
        self._generate_positions()
    
    def _generate_positions(self):

        attempts_limit = 1000  # max tries to place a landmark
        for _ in range(self.num):
            attempts = 0
            while attempts < attempts_limit:
                x = random.randint(self.radius, self.width - self.radius)
                y = random.randint(self.radius, self.height - self.radius)
                pos = (x, y)
                
                if self._is_far_enough(pos):
                    self.positions.append(pos)
                    break
                attempts += 1
            else:
                print(f"Warning: Could only place {_} landmarks out of {self.num} due to spacing constraints.")
                break

    def _is_far_enough(self, pos):
        for p in self.positions:
            dist = math.hypot(pos[0] - p[0], pos[1] - p[1])
            if dist < self.min_dist:
                return False
        return True
    
    def draw(self, win):
        for pos in self.positions:
            pygame.draw.circle(win, self.color, pos, self.radius)
    
    def get_positions(self):
        return self.positions
        

pygame.init()

dim = (1200, 800)
env = Envo(dim)
start = (400, 200)
rob = Robot(start, "Robot.png", 0.01 * 3779.52)
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

run= True

while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()  # get all keys pressed right now
    rob.update_velocities(keys)       # update velocities continuously if keys held
    rob.update_kinematics()           # update position and angle based on velocities

    dt = clock.tick(60) / 1000.0
    prevtime = pygame.time.get_ticks()

    env.win.fill(env.white)           # Clear screen first
    landmarks.draw(env.win)           # Draw landmarks
    rob.draw(env.win)
    env.trajectory((rob.x, rob.y))
    env.write(int(rob.velL), int(rob.velR), rob.theta)
    pygame.display.update()           # Update screen last
    
    sensor.draw(env.win, (rob.x, rob.y), rob.theta)

    #Landmark Detection
    visible_landmarks = sensor.filter_landmarks(landmarks.get_positions(), (rob.x, rob.y), rob.theta)
    
    # For example, highlight visible landmarks in red
    for vpos in visible_landmarks:
        pygame.draw.circle(env.win, env.red, (int(vpos[0]), int(vpos[1])), landmarks.radius + 3, 2)

    pygame.display.update()         

    
pygame.quit()
