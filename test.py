from core.genome_cont import Genome
from core.phenotype_cont import Phenotype
from utils.visualizer import Visualizer

g = Genome()
p = Phenotype(g)

g.change_symbol(0, 0, "p")
g.change_symbol(0, 4, "z167")

while not p.development_finished():
    p.develop()

v = Visualizer()

v.print_phenotype(p)