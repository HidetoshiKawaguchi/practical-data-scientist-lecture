import uuid
import json
import logging
import random
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# すべての warning をコンソール表示、ログファイルに保存する
logging.captureWarnings(True)
warn_logger = logging.getLogger("py.warnings")
warn_logger.setLevel(logging.WARNING)


def exp(
    standarization: bool,
    learning_rate_init: float,
    hidden_layer_sizes: tuple[int, ...],
    activation: str,
    _seed: int
) -> None:
    args_dict = locals()
    hls_str = "-".join(map(str, hidden_layer_sizes))
    param_str = f"a={activation},hls={hls_str},lri={learning_rate_init},s={standarization},_s={_seed}"

    # -----------------------------
    # ロガー設定（コンソール + ファイル）
    # -----------------------------
    logger = logging.getLogger(__name__ + str(uuid.uuid5(uuid.NAMESPACE_DNS, param_str)))
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # コンソール表示は数が多くなるため行わない
    # コンソール
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # ファイル（log を保存）
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{param_str}.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # warn_logger.addHandler(console_handler)
    warn_logger.addHandler(file_handler)

    logger.info(f"=== Experiment Started.===")
    logger.info(f"param = {args_dict}")

    # -----------------------------
    # 乱数シードの初期化
    # -----------------------------
    random.seed(_seed)
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
    pipe_list = []
    if standarization:
        pipe_list.append(("scaler", StandardScaler()))
    pipe_list.append(
        (
            "mlp",
            MLPRegressor(
                learning_rate_init=learning_rate_init,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
            )
        )
    )
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
        "_param": args_dict
    }
    logger.info(f"experiment result = {out_dict}")
    out_dir = Path(__file__).parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"experiment_result_{param_str}.json" 
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved result to {out_path}")

    logger.info("=== Experiment Finished. ===")
    warn_logger.removeHandler(file_handler) # 多重に警告を表示しないために削除


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Run MLPRegressor experiment on wine-quality.")
    p.add_argument("--standarization", dest="standarization",
                   action=BooleanOptionalAction, default=True,
                   help="Enable/disable StandardScaler (default: enabled)")
    p.add_argument("--lri", "--learning_rate_init", dest="learning_rate_init",
                   type=float, default=0.001, help="Learning rate init (default: 0.001)")
    p.add_argument("--hls", "--hidden_layer_sizes", dest="hidden_layer_sizes",
                   type=int, nargs="*", default=[100],
                   help="Hidden layer sizes, e.g. --hls 128 64")
    p.add_argument("--activation", choices=["relu", "tanh", "logistic", "identity"],
                   default="relu", help="Activation function (default: relu)")
    p.add_argument("--seed", dest="_seed", type=int, default=2525,
                   help="Base random seed (default: 2525)")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    exp(
        standarization=args.standarization,
        learning_rate_init=args.learning_rate_init,
        hidden_layer_sizes=tuple(args.hidden_layer_sizes),
        activation=args.activation,
        _seed=args._seed
    )
