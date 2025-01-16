from utils.visualizer import Visualizer

visualizer = Visualizer()

paths = ['logs/lunar_lander.json']

visualizer.plot_avg_fitness(paths, save=False)
