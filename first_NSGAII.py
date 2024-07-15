#!/usr/bin/env python
# REF: https://yuyumoyuyu.com/2021/07/23/howtousepymoo/

import numpy as np
from pymoo.util.misc import stack
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

class MyProblem(Problem):
    """
    Vectorized Evaluation
    """
    def __init__(self):
        super().__init__(
                n_var=2, # 入力変数の次元
                n_obj=2, # 目的関数の数 
                n_constr=2, # 制約条件の数
                xl=np.array([-2, -2]), # 入力変数の下限
                xu=np.array([2, 2]), # 入力変数の上限
                )

    def _evaluate(self, X, out, *args, **kwargs):
        # 目的関数
        f1 = X[:, 0]**2 + X[:, 1]**2
        f2 = (X[:, 0]-1)**2 + X[:, 1]**2
        # 制約条件
        g1 = 2*(X[:, 0]-0.1)*(X[:, 0]-0.9)
        g2 = - 20*(X[:, 0]-0.4)*(X[:, 0]-0.6)
        # 目的関数の値
        out['F'] = np.column_stack([f1, f2])
        # 制約条件の値
        out['G'] = np.column_stack([g1, g2])

# class MyProblem(ElementwizeProblem):
#     """
#     Elementwize Evaluation
#     """
#     def __init__(self):
#         super().__init__(v_var=2, # 入力変数の次元
#                 o_obj=2, # 目的関数の数 
#                 n_constr=2, # 制約条件の数
#                 xl=np.array([-2, 2]), # 入力変数の下限
#                 xu=np.array([2, 2]), # 入力変数の上限
#                 )
# 
#     def __evaluate(self, x, out, *args, **kwargs):
#         # 目的関数
#         f1 = x[0]**2 + x[1]**2
#         f2 = (x[0]-1)**2 + x[1]**2
#         # 制約条件
#         g1 = 2*(x[0]-0.1)*(x[0]-0.9)
#         g2 = 20*(x[0]-0.4)*(x[0]-0.6)
#         # 目的関数の値
#         out['F'] = [f1, f2]
#         # 制約条件の値
#         out['G'] = [g1, g2]


def func_pf(flatten=True, **kwargs):
    """
    パレート解の設定
    パレート解が既知の場合に使用（テスト用）
    """
    f1_a = np.linspace(0.1**2, 0.4**2, 100)
    f2_a = (np.sqrt(f1_a) - 1)**2

    f1_b = np.linspace(0.6**2, 0.9**2, 100)
    f2_b = (np.sqrt(f1_b) - 1)**2

    a = np.column_stack([f1_a, f2_a])
    b = np.column_stack([f1_b, f2_b])
    
    return stack(a, b, flatten=flatten)

def func_ps(flatten=True, **kwargs):
    """
    パレートセット
    """
    x1_a = np.linspace(0.1, 0.4, 50)
    x1_b = np.linspace(0.6, 0.9, 50)
    x2 = np.zeros(50)

    a = np.column_stack([x1_a, x2])
    b = np.column_stack([x1_b, x2])

    return stack(a, b, flatten=flatten)


class MyTestProblem(MyProblem):
    def _calc_pareto_front(self, *args, **kwargs):
        return func_pf(**kwargs)
    def _calc_pareto_set(self, *args, **kwargs):
        return func_ps(**kwargs)


if __name__ == '__main__':
    # 問題の定義
    problem = MyTestProblem()

    # アルゴリズムの初期化（NSGA-IIを使用）
    algorithm = NSGA2(
            pop_size=40, # 集団のサイズ。
            n_offspring=10, # 各世代で生成される子個体数。デフォルトでは親集団と同じサイズ。
            sampling=LHS(), # 初期集団を生成する方法。デフォルトではランダムサンプリング。
            crossover=SBX(prob=0.9, eta=15), # 交叉オペレータ
            mutation=PolynomialMutation(eta=20), # 突然変異オペレータ
            eliminate_duplicates=True # 重複する個体を排除するかどうか。
    )

    # 終了条件（40世代）
    termination = get_termination('n_gen', 40)

    # 最適化の実行
    res = minimize(problem,
            algorithm,
            termination,
            seed=1,
            save_history=True,
            verbose=True)

    # 結果の可視化
    ps = problem.pareto_set(use_cache=False, flatten=False)
    pf = problem.pareto_front(use_cache=False, flatten=False)

    # 目的関数空間
    plot = Scatter(title = 'Objective Space')
    plot.add(res.F)
    if pf is not None:
        plot.add(pf, plot_type='line', color='black', alpha=0.7)
    plot.show()

