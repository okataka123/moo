import scipy.special
import operator
import math
import random

import numpy
import matplotlib.pyplot as plt

from deap import algorithms, base, creator, gp, tools


def target_function(x):
#    return x**4 + x**3 + x**2 + x
   return scipy.special.j1(x)
 
# Define new functions
def protectedDiv(left, right):
   try:
       return left / right
   except ZeroDivisionError:
       return 1
 
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.uniform(-numpy.pi, numpy.pi))
pset.renameArguments(ARG0='x')
 
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
 
previous_champ = "protectedDiv(sin(sin(protectedDiv(x, -1.1291506569801126))), add(mul(x, 0.7731890375960067), sub(-1.1291506569801126, x)))"
 
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
#前の結果をprevious_champにおいて、その結果を読み込ませたいときは
#toolbox.register("expr", gp.PrimitiveTree.from_string, previous_champ, pset=pset)
#とすると全部の遺伝子がこの文字列をprimitiveに変換したものに
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
def evalSymbReg(individual, points):
   # Transform the tree expression in a callable function
   func = toolbox.compile(expr=individual)
   sqerrors = [0 for x in points]
   for i, x in enumerate(points):
       sqe = 1e50
       try:
           fx = func(x)
           sqe = (func(x) - target_function(x))**2
       except:
           print("Overflow!")
           break
       sqerrors.append(sqe)
   return math.fsum(sqerrors) / len(points),
 
toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10, 10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
 
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
 
def main():
   pop = toolbox.population(n=300)
   hof = tools.HallOfFame(1)
 
   stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
   stats_size = tools.Statistics(len)
   mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
   mstats.register("avg", numpy.mean)
   mstats.register("std", numpy.std)
   mstats.register("min", numpy.min)
   mstats.register("max", numpy.max)
 
   pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 2000, stats=mstats,
                                  halloffame=hof, verbose=True)
   # print log
   return pop, log, hof
 
if __name__ == "__main__":
   pop, log, hof = main()
   best = tools.selBest(pop, 1)[0]
   print(best)
