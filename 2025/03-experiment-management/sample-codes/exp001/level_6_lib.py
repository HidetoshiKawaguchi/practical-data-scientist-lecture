import contextlib
import uuid
import json
import logging
import random
from pathlib import Path

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

import numpy as np
import pandas as pd

@contextlib.contextmanager
def experiment_loggers(param_str: str, log_dir: Path, experiment_header: str = "exp"):
    """
    実験ごとに専用 logger と warn_logger を用意する context manager。
    with ブロックを抜けると自動でハンドラを外す。
    """
    # 実験用logger
    logger_name = f"{experiment_header}_{uuid.uuid5(uuid.NAMESPACE_DNS, param_str)}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / f"{experiment_header}_{param_str}.log"
    file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # warning用ロガー
    logging.captureWarnings(True)
    warn_logger = logging.getLogger("py.warnings")
    warn_logger.setLevel(logging.WARNING)
    warn_logger.addHandler(file_handler)

    try:
        yield logger, warn_logger
    finally:
        logger.removeHandler(file_handler)
        warn_logger.removeHandler(file_handler)
        file_handler.close()
        logging.captureWarnings(False)

def init_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(random.randrange(100000000))

def load_Xy() -> (pd.DataFrame, pd.Series):
    csv_path = Path(__file__).parent / "data" / "wine+quality" / "winequality-red.csv"
    df = pd.read_csv(csv_path, sep=";")
    X = df.drop(columns=["quality"])
    y = df["quality"].astype(float)
    return X, y

def evaluate(y_test:np.ndarray, pred: np.ndarray) -> dict[str, float]:    
    return {
        "RMSE": root_mean_squared_error(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "R^2": r2_score(y_test, pred),
    }

def save_out_json(out_path: Path, out_dict: dict[str, any]) -> None:
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=2)
