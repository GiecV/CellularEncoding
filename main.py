from utils.visualizer import Visualizer

visualizer = Visualizer()

paths = ['logs/3i5i.json', 'logs/5i.json']

visualizer.plot_all_runs(paths[0], save=False)
visualizer.plot_all_runs(paths[1], save=False)
visualizer.plot_avg_fitness(paths, save=False)
visualizer.plot_times(paths, save=True)

# paths = ['logs/3i6i.json', 'logs/6i.json']
# visualizer.plot_times(paths, save=True)
# visualizer.plot_avg_fitness(paths, save=True)

# paths = ['logs/log_20241104_132559.json']
# visualizer.print_lineage(paths[0], save=True)
