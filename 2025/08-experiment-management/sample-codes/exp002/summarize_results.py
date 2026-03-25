#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Aggregate SVR experiment JSON files into a CSV with mean/std etc."
    )
    p.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory that contains result JSON files (searched recursively)."
    )
    p.add_argument(
        "--output", "-o", type=str, default="results_summary_svr.csv",
        help="Path to output CSV file."
    )
    p.add_argument(
        "--include-seed", action="store_true",
        help="Include `_seed` in group-by keys (default: exclude)."
    )
    return p.parse_args()


def load_rows(result_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in result_dir.rglob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        params = data.get("_param")
        if not isinstance(params, dict):
            continue

        row: Dict[str, Any] = {
            # metrics
            "RMSE": float(data.get("RMSE")) if data.get("RMSE") is not None else None,
            "MAE": float(data.get("MAE")) if data.get("MAE") is not None else None,
            "R2": float(data.get("R^2")) if data.get("R^2") is not None else None,
            # params (SVR)
            "standarization": bool(params.get("standarization")),
            "kernel": str(params.get("kernel")),
            "C": float(params.get("C")) if params.get("C") is not None else None,
            "epsilon": float(params.get("epsilon")) if params.get("epsilon") is not None else None,
            "_seed": params.get("_seed"),
            # trace
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
        pd.DataFrame().to_csv(args.output, index=False)
        print(f"No JSON files found. Wrote empty CSV to: {Path(args.output).resolve()}")
        return

    df = pd.DataFrame(rows)

    # group keys (seedはデフォ除外)
    group_keys = ["standarization", "kernel", "C", "epsilon"]
    if args.include_seed:
        group_keys.append("_seed")

    agg_map = {
        "RMSE": ["count", "mean", "std", "min", "max"],
        "MAE":  ["count", "mean", "std", "min", "max"],
        "R2":   ["count", "mean", "std", "min", "max"],
    }
    grouped = df.groupby(group_keys, dropna=False).agg(agg_map)

    # flatten columns
    grouped.columns = [
        f"{metric}_{stat}" for metric, stat in grouped.columns.to_flat_index()
    ]
    grouped = grouped.reset_index()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    print(f"Wrote summary CSV to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
