# O-Level Students' Mathematics Score Prediction

## Project Description

This end-to-end machine learning regression project predicts O-Level students' final mathematics examination scores (`final_test`) using demographic, academic, and behavioral attributes to help educators identify students needing extra support. The pipeline compares a naive baseline (`DummyRegressor`), linear models (`LinearRegression`, `Ridge`, `Lasso`), and non-linear ensemble models (`RandomForestRegressor`, `GradientBoostingRegressor`) using cross-validation and a hold-out validation set.

## Repository Structure

```text
O-Level-Students-Mathematics-Score-Prediction/
├── config/config.yaml           # Pipeline configuration, feature lists, seeds, and grids
├── data/student_data.csv        # Raw dataset
├── notebooks/eda.ipynb          # Exploratory Data Analysis with statistical checks
├── src/
│   ├── __init__.py
│   ├── data_preparation.py      # Modular data cleaning, feature engineering, and splitting
│   └── model_training.py        # Model training, GridSearchCV tuning, retraining, and saving
├── tests/test_pipeline.py       # Pytest unit and end-to-end smoke tests
├── artifacts/                   # Auto-generated models, metrics JSON, and test predictions
├── main.py                      # Pipeline entry point with config validation and CLI args
└── requirements.txt             # Pinned Python dependencies
```

## Prerequisites and Installation

* **Prerequisites:** Anaconda or Miniconda (Python 3.11+), Git, and optional VS Code extensions.
* **Installation:**
  ```bash
  git clone https://github.com
  cd O-Level-Students-Mathematics-Score-Prediction
  conda create -n schenv python=3.11 -y && conda activate schenv
  pip install -r requirements.txt
  ```

## Configuration (`config/config.yaml`)

Key parameters include `file_path` (`data/student_data.csv`), `target_column` (`final_test`), `output_dir` (`artifacts`), `random_state` (`42`), `val_test_size` (`0.20`), `val_size` (`0.50`), `cv_folds` (`5`), and `selection_metric` (`R²` with `RMSE` as tie-breaker).

## Executing the Pipeline and Tests

* **Run training pipeline:** `python main.py --config config/config.yaml`
* **Run tests:** `pytest -v`

## Exploratory Data Analysis (EDA) Summary

1. **Dimensions & Cleaning:** 15,900 rows and 18 columns. Removed 763 duplicate records based on `student_id`. Dropped rows missing `final_test` (495 rows) or `attendance_rate` (798 rows). Standardized 3,959 `CCA` missing/variant entries into `'No_CCA'`.
2. **Standardization & Encoding:** Normalized uppercase `CCA` labels to title case and mapped binary columns (`gender`, `tuition`, `direct_admission`) to `1` and `0`.
3. **Age & Feature Engineering:** Corrected age typos (`-5`, `5` to `15`; `6` to `16`) and dropped invalid `-4` entries. Engineered `duration_hours` (adjusting for midnight crossover) and `total_class_size` (`n_male` + `n_female`). Dropped low-signal and identifier columns.

## Pipeline Logical Flow & Model Results

* **Flow:** Validates configuration/schema, cleans and splits data (80% train, 10% val, 10% test), applies `ColumnTransformer` preprocessing, evaluates and tunes models via `GridSearchCV`, retrains the top model on combined train/val data, and evaluates on the test set.
* **Validation & Test Results:** Linear and Ridge regression models achieved an $R^2$ of ~0.5904 on validation data and ~0.5811 ($RMSE = 9.0326$, $MAE = 7.2249$) on the final test set.
