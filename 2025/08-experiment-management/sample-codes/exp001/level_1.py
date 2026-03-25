import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 乱数シードの初期化
random.seed(252525)
np.random.seed(random.randrange(10000000))

# データの準備
print("Data preparation...")
csv_path = Path(__file__).parent / "data" / "wine+quality" / "winequality-red.csv"
df = pd.read_csv(csv_path, sep=";")
X = df.drop(columns=["quality"])
y = df["quality"].astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# モデルの準備・学習
print("Model preparation and training...")
pipe_list = [
    ("scaler", StandardScaler()),
    (
        "mlp",
        MLPRegressor(
            learning_rate_init=0.001,
            hidden_layer_sizes=(100,),  # 1層100ノード
            activation="relu",
        ),
    ),
]
pipe = Pipeline(pipe_list)
pipe.fit(X_train, y_train)

# 予測と評価
print("Prediction and evaluation...")
pred = pipe.predict(X_test)
print(f"RMSE: {root_mean_squared_error(y_test, pred):.4f}")
print(f"MAE : {mean_absolute_error(y_test, pred):.4f}")
print(f"R^2 : {r2_score(y_test, pred):.4f}")
