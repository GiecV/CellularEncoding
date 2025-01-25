import math
from shapely.geometry import LineString, Point

class Robot:
    def __init__(self, start, maze):
        self.position = Point(start)
        self.orientation = 0  # Angle in radians
        self.goal_index = 0
        self.maze = maze
        self.lidar_angles = [-math.pi / 4, 0, math.pi / 4, math.pi / 2, -math.pi / 2]
        self.lidar_readings = []

    def rotate(self, angle):
        self.orientation += angle
        self.orientation %= 2 * math.pi

    def move_forward(self, distance):
        dx = distance * math.cos(self.orientation)
        dy = distance * math.sin(self.orientation)
        new_position = Point(self.position.x + dx, self.position.y + dy)
        if not self.check_collision(new_position):
            self.position = new_position

    def check_collision(self, new_position):
        path = LineString([self.position, new_position])
        collision_happened = any(path.intersects(wall) for wall in self.maze.walls)
        return collision_happened

    def check_goal_reached(self):
        goal = Point(self.maze.goals[self.goal_index])
        if self.position.distance(goal) < 0.2:  # Goal threshold
            self.goal_index += 1
            self.goal_index = min(self.goal_index, len(self.maze.goals) - 1)

        if self.goal_index == len(self.maze.goals) - 1:
            return True
        else: return False

    def update_lidar(self):
        self.lidar_readings = []
        for angle in self.lidar_angles:
            ray_angle = self.orientation + angle
            ray = LineString([
                self.position,
                (
                    self.position.x + math.cos(ray_angle) * 10,  # Max range 10
                    self.position.y + math.sin(ray_angle) * 10
                )
            ])
            distances = [ray.intersection(wall).distance(self.position) for wall in self.maze.walls if ray.intersects(wall)]
            self.lidar_readings.append(min(distances, default=10))

    def plot(self, ax):
        # Plot robot's position
        ax.plot(self.position.x, self.position.y, 'ro', markersize=8)

        # Plot lidar rays
        for angle, distance in zip(self.lidar_angles, self.lidar_readings):
            ray_angle = self.orientation + angle
            end_x = self.position.x + distance * math.cos(ray_angle)
            end_y = self.position.y + distance * math.sin(ray_angle)
            ax.plot([self.position.x, end_x], [self.position.y, end_y], 'r--')

    def get_fitness(self):
        goal = Point(self.maze.goals[self.goal_index])

        if self.goal_index == 0:
            max_distance = 1
        else:
            previous_goal = Point(self.maze.goals[self.goal_index - 1])
            current_goal = Point(self.maze.goals[self.goal_index])
            max_distance = previous_goal.distance(current_goal)

        d = self.position.distance(goal) / max_distance
        
        return (
            10 if self.goal_index == len(self.maze.goals) - 1 else
            self.goal_index + (1-d)
        )