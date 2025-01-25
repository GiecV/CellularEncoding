import torch
import math

from core.phenotype_cont import Phenotype
from core.nn_cont import NNFromGraph
from tasks.maze import Maze
from tasks.robot import Robot
from shapely.geometry import Point

def compute_fitness(individual, n=5):
    # Simulation Parameters
    walls = [((0, 0), (10, 0)), ((10, 0), (10, 10)), ((10, 10), (0, 10)), ((0, 10), (0, 0)),
            ((0, 6), (4, 8)), ((3, 8.3), (4, 4)), ((4, 0), (4, 2)), ((10, 8), (6, 5))]
    start = (1, 1)
    goals = [(1, 2), (7, 1), (5, 6), (8, 9), (2, 9)]

    maze = Maze(walls, goals)
    robot = Robot(start, maze)

    p = Phenotype(individual)
    nn = NNFromGraph(p, inputs=5, outputs=1)

    # Simulate Episode
    num_steps = 80
    for step in range(num_steps):
        # Example control logic (replace with neural network output)
        if robot.goal_index < len(maze.goals):
            robot.update_lidar()
            readings = robot.lidar_readings
            action = nn.forward(torch.tensor(readings, dtype=torch.float32)).detach().numpy()
            if action < 0:
                robot.move_forward(0.5)
                robot.position = Point(round(robot.position.x, 2), round(robot.position.y, 2))
            else:    
                robot.rotate(math.pi/2)

            # maze.plot(robot)

            if robot.check_goal_reached():
                print('Done! Goal Reached!')
                break

    # Compute and print fitness
    fitness = robot.get_fitness()
    fitness = max(fitness, 0.0)
    print(f"Step: {step}, Position: {robot.position}, Fitness: {fitness}")
    return fitness