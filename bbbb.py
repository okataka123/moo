import operator
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

from deap import algorithms, base, creator, gp, tools
from deap.tools import HallOfFame
from graphviz import Source
import pydot

# ターゲット関数の定義
def target_function(x):
#    return x**4 + x**3 + x**2 + x
    return scipy.special.j1(x)

# 適応度と個体の定義
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 遺伝的プログラミングの設定
toolbox = base.Toolbox()
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.renameArguments(ARG0="x")

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

# 評価関数の定義
def evaluate(individual, points):
    func = toolbox.compile(expr=individual)
    sqerrors = ((func(x) - target_function(x))**2 for x in points)
    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evaluate, points=[x/10. for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# メインの遺伝的アルゴリズムの設定
def main():
    random.seed(42)
    pop = toolbox.population(n=300)
    hof = HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats=stats, halloffame=hof, verbose=True)

    return pop, stats, hof

if __name__ == "__main__":
    pop, stats, hof = main()

    # 最良の個体を表示
    best_individual = hof[0]
    print("Best individual:", best_individual)
    print("Best fitness:", best_individual.fitness.values)

    # 最良の個体のツリーを可視化
    nodes, edges, labels = gp.graph(best_individual)

    # experiment
    eqn = str(best_individual)
    print('equation =', eqn)

    # グラフの作成
    dot = Source(gp.graph(best_individual))
    graph = pydot.Dot(graph_type="digraph")

    for node in nodes:
        graph.add_node(pydot.Node(node, label=labels[node]))

    for edge in edges:
        graph.add_edge(pydot.Edge(edge[0], edge[1]))

    # ファイルに保存
    graph.write_png("best_individual_tree.png")

    # 画面に表示
    from PIL import Image
    Image.open("best_individual_tree.png").show()

    # 最良の個体の関数をプロット
    func = toolbox.compile(expr=best_individual)
    #x = np.linspace(-1, 1, 100)
    x = np.linspace(-10, 10, 1000)
    y = [func(i) for i in x]
    y_target = [target_function(i) for i in x]

    plt.plot(x, y, label="Predicted")
    plt.plot(x, y_target, label="Target")
    plt.legend()
    plt.show()

