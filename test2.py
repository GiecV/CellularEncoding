from utils.visualizer import Visualizer

v = Visualizer()

paths = ['logs/lunar_lander_cont.json']

v.plot_avg_fitness(paths, False)