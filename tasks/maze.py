import matplotlib.pyplot as plt
from shapely.geometry import LineString

class Maze:
    def __init__(self, walls, goals):
        self.walls = [LineString(wall) for wall in walls]
        self.goals = goals

    def plot(self, robot):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        # Plot walls
        for wall in self.walls:
            x, y = wall.xy
            ax.plot(x, y, color='black', linewidth=2)

        # Plot goals
        for i, goal in enumerate(self.goals):
            ax.plot(goal[0], goal[1], 'go' if i > robot.goal_index else 'bo', markersize=10)

        # Plot robot
        robot.plot(ax)

        # Configure plot
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.show()