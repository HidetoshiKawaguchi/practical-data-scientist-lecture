from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from level_6_lib import (
    experiment_loggers,
    init_random_seeds,
    load_Xy,
    evaluate,
    save_out_json
)

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

    # 乱数シードの初期化
    init_random_seeds(_seed)
    
    log_dir = Path(__file__).parent / "logs"
    with experiment_loggers(param_str, log_dir, "exp001") as (logger, _):
        logger.info(f"=== Experiment Started.===")
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
        out_dict = evaluate(y_test, pred)
        logger.info(f"experiment result = {out_dict}")
        out_dict["_param"] = args_dict

        # 保存
        out_dir = Path(__file__).parent / "data" / "results"         
        out_path = out_dir / f"experiment_result_{param_str}.json" 
        save_out_json(out_path, out_dict)
        logger.info(f"Saved result to {out_path}")

        logger.info("=== Experiment Finished. ===")


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
