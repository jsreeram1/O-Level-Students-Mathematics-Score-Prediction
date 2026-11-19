import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTraining:
    """Trains, tunes, selects, retrains, and persists regression pipelines for O-Level math scores."""

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer) -> None:
        self.config = config
        self.preprocessor = preprocessor
        self.seed = config["random_state"]

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Defines baseline and non-linear candidate regressors."""
        return {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(random_state=self.seed),
            "lasso": Lasso(random_state=self.seed),
            "random_forest": RandomForestRegressor(random_state=self.seed, n_jobs=-1),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.seed),
        }

    @staticmethod
    def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Computes standardized regression metrics (MAE, MSE, RMSE, R2)."""
        preds = model.predict(X)
        mse = float(mean_squared_error(y, preds))
        return {
            "MAE": float(mean_absolute_error(y, preds)),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y, preds)),
        }

    def train_and_evaluate_baselines(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """Fits default baseline pipelines and evaluates them on the validation set."""
        trained_pipelines: Dict[str, Pipeline] = {}
        val_metrics: Dict[str, Dict[str, float]] = {}

        for name, regressor in self._get_candidate_models().items():
            logger.info("Training baseline model: %s", name)
            pipe = Pipeline(steps=[
                ("preprocessor", clone(self.preprocessor)),
                ("regressor", regressor),
            ])
            pipe.fit(X_train, y_train)
            metrics = self.evaluate_model(pipe, X_val, y_val)
            trained_pipelines[name] = pipe
            val_metrics[name] = metrics
            logger.info("Validation [%s] -> R2: %.4f | RMSE: %.4f", name, metrics["R2"], metrics["RMSE"])

        return trained_pipelines, val_metrics

    def tune_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]], Dict[str, Any]]:
        """Runs GridSearchCV for models configured with hyperparameter grids."""
        param_grids = self.config.get("param_grids", {})
        cv_folds = self.config.get("cv_folds", 5)
        candidates = self._get_candidate_models()

        tuned_pipelines: Dict[str, Pipeline] = {}
        tuned_val_metrics: Dict[str, Dict[str, float]] = {}
        tuning_summary: Dict[str, Any] = {}

        for name, grid in param_grids.items():
            if name not in candidates:
                logger.warning("Skipping unknown model '%s' in param_grids", name)
                continue

            logger.info("Tuning hyperparameters for: %s", name)
            pipe = Pipeline(steps=[
                ("preprocessor", clone(self.preprocessor)),
                ("regressor", candidates[name]),
            ])

            search = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                cv=cv_folds,
                scoring="r2",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)

            best_pipe = search.best_estimator_
            val_scores = self.evaluate_model(best_pipe, X_val, y_val)

            tuned_pipelines[f"{name}_tuned"] = best_pipe
            tuned_val_metrics[f"{name}_tuned"] = val_scores
            tuning_summary[f"{name}_tuned"] = {
                "best_params": search.best_params_,
                "best_cv_r2": float(search.best_score_),
                "validation_metrics": val_scores,
            }
            logger.info(
                "Tuned [%s] -> Best Params: %s | Val R2: %.4f | Val RMSE: %.4f",
                name, search.best_params_, val_scores["R2"], val_scores["RMSE"]
            )

        return tuned_pipelines, tuned_val_metrics, tuning_summary

    def select_best_model(
        self, all_pipelines: Dict[str, Pipeline], all_metrics: Dict[str, Dict[str, float]]
    ) -> Tuple[str, Pipeline]:
        """Selects the winning model using R2 (highest) with RMSE (lowest) as tie-breaker."""
        if not all_pipelines or not all_metrics:
            raise ValueError("No trained models available for selection.")

        best_name = max(
            all_metrics.keys(),
            key=lambda m: (round(all_metrics[m]["R2"], 6), -all_metrics[m]["RMSE"])
        )
        logger.info(
            "Selected best model: %s (Val R2: %.4f, Val RMSE: %.4f)",
            best_name, all_metrics[best_name]["R2"], all_metrics[best_name]["RMSE"]
        )
        return best_name, all_pipelines[best_name]

    def retrain_and_test_final_model(
        self,
        best_pipeline: Pipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
        """Retrains the winning pipeline on Train + Validation data and evaluates once on Test."""
        X_train_val = pd.concat([X_train, X_val], axis=0)
        y_train_val = pd.concat([y_train, y_val], axis=0)

        final_pipeline = clone(best_pipeline)
        final_pipeline.fit(X_train_val, y_train_val)

        test_metrics = self.evaluate_model(final_pipeline, X_test, y_test)
        test_preds = final_pipeline.predict(X_test)

        predictions_df = pd.DataFrame({
            "actual_score": y_test.values,
            "predicted_score": test_preds,
            "residual": y_test.values - test_preds,
        })

        logger.info("Final Test Evaluation -> %s", test_metrics)
        return final_pipeline, test_metrics, predictions_df

    def save_artifacts(
        self,
        final_pipeline: Pipeline,
        best_model_name: str,
        val_metrics: Dict[str, Dict[str, float]],
        tuning_summary: Dict[str, Any],
        test_metrics: Dict[str, float],
        predictions_df: pd.DataFrame,
    ) -> str:
        """Persists fitted pipeline, metrics, predictions, and config snapshot to disk."""
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, "best_math_score_pipeline.joblib")
        joblib.dump(final_pipeline, model_path)

        predictions_path = os.path.join(output_dir, "test_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

        config_snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
        with open(config_snapshot_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f)

        manifest = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "selected_model": best_model_name,
            "random_state": self.seed,
            "environment": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "validation_comparison": val_metrics,
            "hyperparameter_tuning": tuning_summary,
            "final_test_metrics": test_metrics,
        }

        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        logger.info("All model artifacts saved to '%s'", output_dir)
        return model_path

    @staticmethod
    def predict_new_data(model_path: str, cleaned_features: pd.DataFrame) -> np.ndarray:
        """Loads a saved pipeline artifact and predicts scores on new student records."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Saved model not found at: {model_path}")
        pipeline = joblib.load(model_path)
        return pipeline.predict(cleaned_features)
