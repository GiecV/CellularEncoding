from utils.visualizer import Visualizer

v = Visualizer()
paths = ['logs/gates_up_to_n.json']

# v.create_boxplots(paths, save=False)
# v.create_sum_boxplots(paths, save=False)

v.save_best(paths[0], show=False)
