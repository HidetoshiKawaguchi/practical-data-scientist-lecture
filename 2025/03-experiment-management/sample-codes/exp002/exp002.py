import random
import math
from itertools import product

from tqdm import tqdm

# SVR 版の exp（先ほど作ったもの）
from svr_exp import exp

if __name__ == "__main__":
    n_random_seed = 3  # 各パラメータ組み合わせにつき何回走らせるか

    # ---- SVR のハイパーパラメータグリッド ----
    param_grid = {
        # SVR は標準化した方が安定しやすいので True 推奨
        "standarization": [True],
        "kernel": ["rbf", "linear"],  # 必要に応じて絞ってOK
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.1, 0.05],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    total = math.prod(len(v) for v in values)  # 全組み合わせ数

    for combo in tqdm(product(*values), total=total, desc="Grid(SVR)"):
        params = dict(zip(keys, combo))
        for _ in range(n_random_seed):
            params["_seed"] = random.randrange(1_000_000)
            exp(**params)
