import operator
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms, base, creator, gp, tools

# 1. ターゲット関数の定義
def target_function(x):
    return x**4 + x**3 + x**2 + x

# 2. 適応度と個体の定義
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# 3. 遺伝的プログラミングの設定
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

# 4. 評価関数の定義
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

# 5. メインの遺伝的アルゴリズムの設定
def main():
    random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
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

    # 最良の個体の関数をプロット
    func = toolbox.compile(expr=best_individual)
    x = np.linspace(-1, 1, 100)
    y = [func(i) for i in x]
    y_target = [target_function(i) for i in x]

    plt.plot(x, y, label="Predicted")
    plt.plot(x, y_target, label="Target")
    plt.legend()
    plt.show()

"""
プログラムの説明

    1. ターゲット関数の定義: target_function で、シンボル回帰によって推測したい関数を定義します。
    2. 適応度と個体の定義: FitnessMin と Individual クラスを作成し、最小化問題として設定します。
    3. 遺伝的プログラミングの設定: pset で、使用する基本演算子（加算、減算、乗算、負数、三角関数）や定数を定義します。
    4. 評価関数の定義: evaluate で、個体の適応度を計算します。適応度は予測された値とターゲット関数の値の二乗誤差の平均として計算されます。
    5. メインの遺伝的アルゴリズムの設定: main で、遺伝的アルゴリズムを実行し、最良の個体をホールオブフェームに記録します。

このコードを実行すると、最適な関数の形状を見つけるために進化的アルゴリズムを使用し、最良の個体の関数をプロットします。

それぞれの行の意味について

1. toolbox = base.Toolbox()
    DEAPで使用するツールボックスを作成します。ツールボックスには遺伝的アルゴリズムの各種操作
    （個体生成、交叉、突然変異、選択など）を登録します。

2. pset = gp.PrimitiveSet("MAIN", 1)
    遺伝的プログラミング用のプリミティブセットを作成します。このセットには使用する関数や端末
    （変数や定数）を定義します。ここでは、1つの引数（変数x）を持つメインセットを作成します。

3. pset.addPrimitive(operator.add, 2)
    プリミティブセットに二項加算関数（加法、引数2つ）を追加します。
    operator.addは2つの引数を受け取り、その和を返す関数です。

4. pset.addPrimitive(operator.sub, 2)
    プリミティブセットに二項減算関数（減法、引数2つ）を追加します。
    operator.subは2つの引数を受け取り、その差を返す関数です。

5. pset.addPrimitive(operator.mul, 2)
    プリミティブセットに二項乗算関数（乗法、引数2つ）を追加します。
    operator.mulは2つの引数を受け取り、その積を返す関数です。

6. pset.addPrimitive(operator.neg, 1)
    プリミティブセットに単項否定関数（符号反転、引数1つ）を追加します。
    operator.negは1つの引数を受け取り、その符号を反転させた値を返す関数です。

7. pset.addPrimitive(math.cos, 1)
    プリミティブセットに単項余弦関数（コサイン、引数1つ）を追加します。
    math.cosは1つの引数を受け取り、その余弦を返す関数です。

8. pset.addPrimitive(math.sin, 1)
    プリミティブセットに単項正弦関数（サイン、引数1つ）を追加します。
    math.sinは1つの引数を受け取り、その正弦を返す関数です。

9. pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
    プリミティブセットにエフェメラル定数（ランダム定数）を追加します。ここでは、
    lambda: random.randint(-1, 1)により-1から1の間のランダムな整数を返す関数を定義します。

10. pset.renameArguments(ARG0="x")
    プリミティブセットの引数名をデフォルトのARG0からxに変更します。
    これにより、生成される数式の引数がxとして表現されます。

11. toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    gp.genHalfAndHalfメソッドを使用して、最小深度1、最大深度2の半々生成法による式（表現）を
    生成する操作をtoolboxに登録します。これにより、toolbox.expr()を呼び出すと、プリミティブセットpsetを基にした新しい表現が生成されます。

12. toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    tools.initIterateを使用して、toolbox.expr関数で生成される表現を持つ新しい個体（creator.Individual）を
    生成する操作をtoolboxに登録します。これにより、toolbox.individual()を呼び出すと、新しい個体が生成されます。

13. toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    tools.initRepeatを使用して、toolbox.individual関数で生成される個体のリスト（集団）を
    生成する操作をtoolboxに登録します。これにより、toolbox.population(n)を呼び出すと、n個の新しい個体からなる集団が生成されます。

14. toolbox.register("compile", gp.compile, pset=pset)
    gp.compileを使用して、プリミティブセットpsetを基に個体を関数に変換する操作をtoolboxに登録します。
    これにより、toolbox.compile(expr)を呼び出すと、expr（表現）を基にしたPython関数が生成されます。

15. toolbox.register("evaluate", evaluate, points=[x/10. for x in range(-10, 10)])
    evaluate関数を使用して、pointsリストの点での誤差を計算する評価操作をtoolboxに登録します。
    これにより、toolbox.evaluate(individual)を呼び出すと、指定された点での個体の適応度が計算されます。

16. toolbox.register("select", tools.selTournament, tournsize=3)
    トーナメント選択（選択操作）をtoolboxに登録します。tournsize=3は、3個体のトーナメントを使用することを示します。
    これにより、toolbox.select(population, k)を呼び出すと、k個の個体が選択されます。

17. toolbox.register("mate", gp.cxOnePoint)
    1点交叉操作をtoolboxに登録します。これにより、toolbox.mate(parent1, parent2)を呼び出すと、
    2つの親個体の間で1点交叉が行われ、新しい子個体が生成されます。

18. toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    gp.genFullを使用して、最小深度0、最大深度2の完全生成法による新しい表現を生成する操作をtoolboxに登録します。
    これにより、toolbox.expr_mut()を呼び出すと、新しい表現が生成されます。

19. toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    一様突然変異操作をtoolboxに登録します。この操作では、指定された確率で個体の一部が新しい表現
    （toolbox.expr_mutによって生成される）に置き換えられます。これにより、toolbox.mutate(individual)を呼び出すと、個体が突然変異します。

20. toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    gp.staticLimitデコレータを使用して、交叉操作の結果として生成される個体の高さを最大17に制限します。
    これにより、toolbox.mateによって生成される個体の高さが17を超えないように制限されます。

21. toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    gp.staticLimitデコレータを使用して、突然変異操作の結果として生成される個体の高さを最大17に制限します。
    これにより、toolbox.mutateによって生成される個体の高さが17を超えないように制限されます。
"""
