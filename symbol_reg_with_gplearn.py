import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# 1. ターゲット関数の定義
def target_function(x):
    return x**4 + x**3 + x**2 + x

# 2. データの生成
X = np.linspace(-1, 1, 100).reshape(-1, 1)
y = target_function(X).ravel()

# 3. シンボル回帰モデルの設定
model = SymbolicRegressor(
    population_size=1000,  # 個体群のサイズ
    generations=20,        # 世代数
    tournament_size=20,    # トーナメントのサイズ
    stopping_criteria=0.01, # 停止基準（適応度の変化がこの値以下で停止）
    const_range=(0, 1),    # 定数の範囲
    init_depth=(2, 6),     # 初期個体の深さ
    init_method='half-and-half', # 初期個体群の生成方法
    function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos'), # 使用する演算子
    metric='mean absolute error', # 評価指標
    random_state=42          # 乱数シード
)

# 4. モデルの学習
model.fit(X, y)

# 5. 最良の個体の表示
print("Best individual:", model._best_program)

# 6. 予測とプロット
y_pred = model.predict(X)

plt.figure(figsize=(10, 6))
plt.plot(X, y, label='True function', color='blue')
plt.plot(X, y_pred, label='GP prediction', color='red', linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Symbolic Regression with gplearn')
plt.show()

"""
プログラムの説明

    ターゲット関数の定義:
        target_function で、回帰対象となる関数を定義します。

    データの生成:
        X に入力値を、y にターゲット関数に基づく出力値を生成します。

    シンボル回帰モデルの設定:
        SymbolicRegressor を使用して、遺伝的プログラミングモデルを設定します。主要なパラメータには、個体群のサイズ、世代数、トーナメントサイズ、定数の範囲、使用する関数のセットなどがあります。

    モデルの学習:
        fit メソッドで、データに基づいてシンボル回帰を行います。

    最良の個体の表示:
        model._best_program で、学習過程で見つけた最良の関数（プログラム）を表示します。

    予測とプロット:
        predict メソッドで予測を行い、元のデータと予測結果をプロットします。

実行結果

このコードを実行すると、以下のような出力とプロットが得られます：

    コンソール出力: 学習によって見つけた最良の関数が表示されます。
    グラフ: ターゲット関数とシンボル回帰モデルによって予測された関数が比較されます。

gplearnを使うことで、進化的に最適化された数式を用いてデータにフィットする関数を見つけることができます。
"""
