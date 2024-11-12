from utils.visualizer import Visualizer

visualizer = Visualizer()

# paths = ['logs/3i5i.json', 'logs/5i.json']

# visualizer.plot_all_runs(paths[0], save=True)
# visualizer.plot_all_times(paths[0], save=True)
# visualizer.plot_all_runs(paths[1], save=True)
# visualizer.plot_all_times(paths[1], save=True)
# visualizer.plot_avg_fitness(paths, save=True)
# visualizer.plot_times(paths, save=True)

# visualizer.save_lineage(paths[0], show=False)

paths = ['logs/6i.json', 'logs/36i.json', 'logs/23456i.json']
# visualizer.plot_all_runs(paths[2], save=True)
# visualizer.plot_all_times(paths[2], save=True)
# visualizer.plot_avg_fitness(paths, save=True)
# visualizer.plot_times(paths, save=True)

# visualizer.create_boxplots(paths, save=False)
visualizer.save_best_networks(paths[2], show=False)
