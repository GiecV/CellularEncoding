from utils.visualizer import Visualizer

visualizer = Visualizer()

# paths = ['logs/6i.json', 'logs/36i.json', 'logs/23456i.json']
# paths = ['logs/log_20241121_145821.json']
paths =['logs/acrobot.json']

# visualizer.create_boxplots(paths, save=False)
# visualizer.save_best_networks(paths[0], show=False)
# visualizer.save_best_networks(paths[1], show=False)
# visualizer.save_best_networks(paths[3], show=False)
visualizer.create_boxplots(paths, save=False)
