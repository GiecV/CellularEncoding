from tasks.maze import Maze
from tasks.robot import Robot
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.phenotype import Phenotype
from core.nn_cont import NNFromGraph

steps = 500

walls = [((0, 0), (10, 0)), ((10, 0), (10, 10)), ((10, 10), (0, 10)), ((0, 10), (0, 0)),
         ((0,6), (4,9)), ((3, 8.3), (4, 4)), ((4, 0), (4, 2)), ((10, 8), (6, 5))]
start = (1, 1)
goals = [(1, 2), (7, 1), (5, 6), (8, 9), (2, 9)]
maze = Maze(walls, start, goals)
robot = Robot(maze, start)


finished = False

readings = robot.measure()

maze.plot(robot)
if robot.reached_goal():
    finished = robot.update_goal()

    
print( (
    10
    if robot.goal_index == len(robot.maze.goals) - 1
    else robot.goal_index + max(1 - robot.position.distance(robot.goal), 0)
))