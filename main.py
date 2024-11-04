from utils.visualizer import Visualizer

visualizer = Visualizer()
# visualizer.plot_all_runs('logs/3i6i.json', save=False)
# visualizer.plot_average_fitness('logs/3i6i.json', save=True)
# visualizer.plot_time('logs/3i6i.json', save=False)

# paths = ['logs/3i6i.json', 'logs/6i.json']
# visualizer.plot_times(paths, save=True)
# visualizer.plot_avg_fitness(paths, save=True)

paths = ['logs/log_20241104_132559.json']
visualizer.print_lineage(paths[0], save=True)
