# 現場最前線から教えるデータサイエンス 講義資料

このリポジトリは、2025年度にとある大学で実施した講義の資料・演習用コード・サンプルコードを公開するためのものです。大学からの要請により、大学名および講義名は伏せており、本 README では講義名を「現場最前線から教えるデータサイエンス」として記載しています。

資料は教育目的で公開しています。内容は川口個人の見解であり、所属組織の公式見解を表すものではありません。

なお、ディレクトリ名や資料名の番号は各回の講義順に対応しています。

## この講義で扱うこと

本講義では、人工知能や機械学習そのものの最新理論を深掘りするというより、データサイエンティストとして実務で成果を出すための基礎的かつ実践的な技能を扱います。特に、以下のようなテーマに焦点を当てています。

- ITベンダーにおけるデータサイエンティストの役割
- ドキュメント・コミュニケーション
- 分析・実験・検証の管理方法
- Python を用いた実践的な開発技術

## リポジトリ構成

```text
.
└── 2025
    ├── 01-course-overview
    │   └── 01-course-overview.pdf
    ├── 02-document-communication
    │   └── 02-document-communication.pdf
    ├── 03-experiment-management
    │   ├── 03-experiment-management.pdf
    │   ├── exercises
    │   └── sample-codes
    └── 04-software-engineering
        ├── 04-software-engineering.pdf
        ├── exercises
        └── sample-codes
```

## 各回の資料

### 01 ITベンダーにおけるデータサイエンティスト

`2025/01-course-overview/01-course-overview.pdf`

講義全体の導入回です。ITベンダーにおけるデータサイエンティスト像、周辺ロールとの違い、Python のバージョン管理と仮想環境管理の基礎を扱います。後続回で使う `pyenv` と `venv` の前提もここで説明しています。

### 02 ドキュメント・コミュニケーション

`2025/02-document-communication/02-document-communication.pdf`

データサイエンティストの主要な成果物を「ドキュメント」と捉え、ビジネス文書の種類、伝わる資料の考え方、ストーリー構成、パラグラフ・ライティングの基礎を扱います。

### 03 分析・実験・検証の実践的管理方法

`2025/03-experiment-management/03-experiment-management.pdf`

CRISP-DM、EDA と実験の位置づけ、Jupyter Notebook の長所と限界、実験コードを再現性・追跡性・再利用性の観点からどう改善していくかを扱います。段階的に悪い実験コードを改善していくサンプルが同梱されています。

### 04 データサイエンティストの開発技術

`2025/04-software-engineering/04-software-engineering.pdf`

Python の型ヒント、PEP 8、`mypy`、`flake8`、`black`、`isort`、`pytest` を題材に、実務で最低限必要になるソフトウェアエンジニアリングの基礎を扱います。

## 演習・サンプルコード

### 03 の演習とサンプル

`2025/03-experiment-management/exercises`

- `level_0.py` は改善前の実験コードです。
- `requirements.txt` は演習用の最小依存関係です。
- `data/wine+quality/` には演習用データが含まれています。

`2025/03-experiment-management/sample-codes`

- `README.md` は実験ディレクトリの基本ルールを示しています。
- `exp001/` は MLPRegressor を題材に、`level_0.py` から `level_6.py` まで段階的に実験コードを改善するサンプルです。
- `exp001/eda_wine-quality.ipynb` には赤ワインデータに対する初歩的な EDA がまとまっています。
- `exp001/summarize_results.py` は JSON 形式の実験結果を CSV に集計するスクリプトです。
- `exp001/data/results_summary_mlp.csv` には集計済みの実験結果が含まれます。
- `exp002/` は `exp001` の部品を再利用し、SVR による継続実験を行うサンプルです。
- `exp002/data/results_summary_svr.csv` には SVR 実験の集計済み結果が含まれます。

### 04 の演習とサンプル

`2025/04-software-engineering/exercises`

- `exercise_1_1.py` から `exercise_1_3.py` は型ヒント・規約・整形の演習用ファイルです。
- `exercise_2_1_test.py` から `exercise_2_3_test.py` は `pytest` の穴埋め演習です。
- `requirements.txt` には `mypy`、`flake8`、`pytest` など演習用ツールが含まれます。

`2025/04-software-engineering/sample-codes`

- 型ヒント、`mypy`、`flake8`、`black`、`isort` の動作例を示す単体ファイルがあります。
- `test_sample.py` は最小の `pytest` 例です。
- `mypackage/` には本体コードと `tests/` を分けた pytest の基本構成例があります。

## データについて

03回目の演習・サンプルでは UCI Machine Learning Repository の Wine Quality データセットを利用しています。リポジトリ内には以下が含まれています。

- 赤ワインデータ `winequality-red.csv`
- 白ワインデータ `winequality-white.csv`
- データセット説明 `winequality.names`

`winequality.names` に記載の通り、赤ワインは 1,599 件、白ワインは 4,898 件で、11 個の入力変数と品質ラベルを含みます。講義中の EDA と実験は主に赤ワインデータを対象にしています。

## 実行環境の目安

このリポジトリには複数の独立した教材が含まれているため、環境構築方法はディレクトリごとに異なります。

- 03回目の演習は `requirements.txt` ベースで実行できます。
- 03回目のサンプルコードは `poetry` を前提にした `pyproject.toml` と `poetry.lock` を含みます。
- 04回目の演習も `requirements.txt` ベースです。
- 04回目のサンプルコードには `poetry` ベースの設定が含まれます。

Python は 3.11 系を前提にしている箇所が多く、講義資料でも `pyenv` と仮想環境の利用を推奨しています。

## このリポジトリの読み方

初めて見る場合は、以下の順に読むと意図が追いやすいです。

1. `2025/01-course-overview/01-course-overview.pdf`
2. `2025/02-document-communication/02-document-communication.pdf`
3. `2025/03-experiment-management/03-experiment-management.pdf`
4. `2025/04-software-engineering/04-software-engineering.pdf`
5. 必要に応じて各回の `exercises/` と `sample-codes/`

## 注意事項

- 講義資料は授業で使用した内容をほぼそのまま公開しています。
- 一部のコードや記述は教育用に単純化しています。
- 実験サンプルは「実務でそのまま使う完成形」ではなく、改善の考え方を学ぶための教材です。
- ロックファイルや集計済み CSV は、当時の講義用環境・結果の記録として残しています。
