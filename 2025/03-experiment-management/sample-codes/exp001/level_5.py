import random
from itertools import product

from tqdm import tqdm

from level_5_exp import exp

if __name__ == "__main__":
    n_random_seed = 3
    param_grid = {
        "standarization": [True, False],
        "learning_rate_init": [0.001, 0.01, 0.1],
        "hidden_layer_sizes": [(100, ), (1000, 100), (10,)],
        "activation": ["relu", "tanh", "logistic", "identity"],
    }
    keys = param_grid.keys()
    values = param_grid.values()
    for combo in tqdm(product(*values), total=len(list(product(*values)))):
        params = dict(zip(keys, combo))
        for _ in range(3):
            _seed = random.randrange(1000000)
            params["_seed"] = _seed
            exp(**params)
    
