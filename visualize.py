from utils.visualizer import Visualizer

v = Visualizer()
paths = ['logs/stepping_gates_up_to_n_updated.json']

v.create_boxplots(paths, save=False)
# v.create_sum_boxplots(paths, save=False)

# v.save_best(paths[0], show=False)
