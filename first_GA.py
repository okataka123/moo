import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# 1. 問題の定義
class MyProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=1, n_constr=0, xl=np.array([-5, -5]), xu=np.array([5, 5]))

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = x[0]**2 + x[1]**2
        out["F"] = f1

# 2. 問題のインスタンスを作成
problem = MyProblem()

# 3. アルゴリズムの選択と設定
algorithm = GA(pop_size=20, eliminate_duplicates=True) # 各世代で20個体(解)を保持する。

# 4. 最適化の実行
termination = get_termination("n_gen", 100) # 100世代まわす
res = minimize(problem, algorithm, termination, seed=1, verbose=True)

# 5. 結果の表示
print("最適化された設計変数: %s" % res.X)
print("最小化された目的関数値: %s" % res.F)
