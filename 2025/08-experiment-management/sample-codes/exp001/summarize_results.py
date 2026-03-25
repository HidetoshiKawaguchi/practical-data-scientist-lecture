#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate experiment JSON files into a CSV with mean/std etc."
    )
    p.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory that contains result JSON files (searched recursively)."
    )
    p.add_argument(
        "--output", "-o", type=str, default="results_summary.csv",
        help="Path to output CSV file."
    )
    p.add_argument(
        "--include-seed", action="store_true",
        help="Include `_seed` in group-by keys (default: exclude)."
    )
    return p.parse_args()


def _hls_to_str(hls: Any) -> str:
    """
    hidden_layer_sizes を '100-50-10' のような文字列に正規化
    JSONが [100, 50] / (100,50) / 100 などでも頑健に処理
    """
    if hls is None:
        return "none"
    if isinstance(hls, (list, tuple)):
        return "-".join(str(int(x)) for x in hls)
    # 単一の数値が来たケース
    try:
        return str(int(hls))
    except Exception:
        return str(hls)


def load_rows(result_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in result_dir.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # 壊れたJSONなどはスキップ
            continue

        # 期待するキーがなければスキップ
        if "_param" not in data:
            continue

        params = data.get("_param", {})
        row: Dict[str, Any] = {
            # メトリクス（float化しておく）
            "RMSE": float(data.get("RMSE")) if data.get("RMSE") is not None else None,
            "MAE": float(data.get("MAE")) if data.get("MAE") is not None else None,
            "R2": float(data.get("R^2")) if data.get("R^2") is not None else None,
            # パラメータ（型を整える）
            "standarization": bool(params.get("standarization")),
            "learning_rate_init": float(params.get("learning_rate_init")),
            "hidden_layer_sizes": _hls_to_str(params.get("hidden_layer_sizes")),
            "activation": str(params.get("activation")),
            "_seed": params.get("_seed"),
            # 追跡用
            "_source_file": str(path),
        }
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input)
    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    rows = load_rows(in_dir)
    if not rows:
        # 空でもCSVは出しておくとワークフロー上親切
        pd.DataFrame().to_csv(args.output, index=False)
        print(f"No JSON files found. Wrote empty CSV to: {args.output}")
        return

    df = pd.DataFrame(rows)

    # 集計キー
    group_keys = [
        "standarization",
        "learning_rate_init",
        "hidden_layer_sizes",
        "activation",
    ]
    if args.include_seed:
        group_keys.append("_seed")

    # 集計（count, mean, std, min, max）
    agg_map = {
        "RMSE": [ "mean", "std", "min", "max"],
        "MAE":  [ "mean", "std", "min", "max"],
        "R2":   [ "mean", "std", "min", "max"],
    }
    grouped = df.groupby(group_keys, dropna=False).agg(agg_map)

    # 列名を平坦化
    grouped.columns = [
        f"{metric}_{stat}" for metric, stat in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    # CSV 出力
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    print(f"Wrote summary CSV to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
