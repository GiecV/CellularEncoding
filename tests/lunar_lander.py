import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from utils.visualizer import Visualizer

visualizer = Visualizer()

paths = ['logs/lunar_lander.json']

visualizer.plot_avg_fitness(paths, save=False)
