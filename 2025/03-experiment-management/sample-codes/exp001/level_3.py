import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------
# ロガー設定（コンソール + ファイル）
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# コンソール
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ファイル（log を保存）
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_dir / "experiment.log", mode="a", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# すべての warning をコンソール表示、ログファイルに保存する
logging.captureWarnings(True)
warn_logger = logging.getLogger("py.warnings")
warn_logger.setLevel(logging.WARNING)
warn_logger.addHandler(console_handler)
warn_logger.addHandler(file_handler)

logger.info("=== Experiment Started. ===")

# -----------------------------
# 乱数シードの初期化
# -----------------------------
random.seed(252525)
np.random.seed(random.randrange(10000000))

# -----------------------------
# データの準備
# -----------------------------
logger.info("Data preparation...")
csv_path = Path(__file__).parent / "data" / "wine+quality" / "winequality-red.csv"
df = pd.read_csv(csv_path, sep=";")
X = df.drop(columns=["quality"])
y = df["quality"].astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# モデルの準備・学習
# -----------------------------
logger.info("Model preparation and training...")
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

# -----------------------------
# 予測と評価
# -----------------------------
logger.info("Prediction and evaluation...")
pred = pipe.predict(X_test)
out_dict = {
    "RMSE": root_mean_squared_error(y_test, pred),
    "MAE": mean_absolute_error(y_test, pred),
    "R^2": r2_score(y_test, pred),
}
logger.info(f"experiment result = {out_dict}")
out_dir = Path(__file__).parent / "data" / "results"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "experiment_result.json" 
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved result to {out_path}")

logger.info("=== Experiment Finished. ===")
