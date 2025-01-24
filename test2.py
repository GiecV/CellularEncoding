from utils.visualizer import Visualizer

v = Visualizer()

paths = ['logs/parity_up_to_n.json']

v.create_boxplots(paths)
v.create_sum_boxplots(paths)