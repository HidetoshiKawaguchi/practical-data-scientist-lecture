from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from exp001.level_6_lib import (
    experiment_loggers,
    init_random_seeds,
    load_Xy,
    evaluate,
    save_out_json
)


def exp(
    standarization: bool,
    kernel: str,
    C: float,
    epsilon: float,
    _seed: int,
) -> None:
    args_dict = locals().copy()
    param_str = f"svr_k={kernel},C={C},eps={epsilon},s={standarization},_s={_seed}"

    # 乱数シードの初期化
    init_random_seeds(_seed)

    log_dir = Path(__file__).parent / "logs"
    with experiment_loggers(param_str, log_dir, "exp002") as (logger, _):
        logger.info("=== Experiment Started. ===")
        logger.info(f"param = {args_dict}")

        # -----------------------------
        # データの準備
        # -----------------------------
        logger.info("Data preparation...")
        X, y = load_Xy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -----------------------------
        # モデルの準備・学習 (SVR)
        # -----------------------------
        logger.info("Model preparation and training (SVR)...")
        steps = []
        if standarization:
            steps.append(("scaler", StandardScaler()))
        steps.append((
            "svr",
            SVR(kernel=kernel, C=C, epsilon=epsilon),
        ))
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)

        # -----------------------------
        # 予測と評価
        # -----------------------------
        logger.info("Prediction and evaluation...")
        pred = pipe.predict(X_test)
        out_dict = evaluate(y_test, pred)
        logger.info(f"experiment result = {out_dict}")
        out_dict["_param"] = args_dict

        # 保存
        out_dir = Path(__file__).parent / "data" / "results"
        out_path = out_dir / f"svr_{param_str}.json"
        save_out_json(out_path, out_dict)
        logger.info(f"Saved result to {out_path}")

        logger.info("=== Experiment Finished. ===")


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Run SVR experiment on wine-quality.")
    p.add_argument("--standarization", dest="standarization",
                   action=BooleanOptionalAction, default=True,
                   help="Enable/disable StandardScaler (default: enabled)")

    p.add_argument("--kernel", type=str, default="rbf",
                   help="SVR kernel (default: rbf)")
    p.add_argument("--C", type=float, default=1.0,
                   help="SVR C parameter (default: 1.0)")
    p.add_argument("--epsilon", type=float, default=0.1,
                   help="SVR epsilon (default: 0.1)")

    p.add_argument("--seed", dest="_seed", type=int, default=2525,
                   help="Base random seed (default: 2525)")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    exp(
        standarization=args.standarization,
        kernel=args.kernel,
        C=args.C,
        epsilon=args.epsilon,
        _seed=args._seed,
    )
