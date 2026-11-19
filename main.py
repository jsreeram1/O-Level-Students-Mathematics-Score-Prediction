import argparse
import logging
import sys
from typing import Dict, Any

import yaml

from src.data_preparation import DataPreparation
from src.model_training import ModelTraining

REQUIRED_CONFIG_KEYS = [
    "file_path",
    "target_column",
    "output_dir",
    "random_state",
    "val_test_size",
    "val_size",
    "numerical_features",
    "nominal_features",
    "passthrough_features",
]


def setup_logging() -> None:
    """Configures timestamped logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Loads YAML config and verifies all mandatory keys are present."""
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as exc:
        logging.error("Configuration file not found: %s", config_path)
        raise exc
    except yaml.YAMLError as exc:
        logging.error("Invalid YAML syntax in %s: %s", config_path, exc)
        raise exc

    if not isinstance(config, dict):
        raise ValueError("Configuration file did not parse into a valid dictionary.")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    return config


def run_training_pipeline(config_path: str) -> None:
    """Executes the full end-to-end training, tuning, evaluation, and artifact-saving workflow."""
    logger = logging.getLogger(__name__)
    config = load_and_validate_config(config_path)

    data_prep = DataPreparation(config)
    raw_df = data_prep.load_data(config["file_path"])
    cleaned_df = data_prep.clean_data(raw_df, is_training=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.split_data(cleaned_df)

    trainer = ModelTraining(config, data_prep.preprocessor)

    baseline_models, baseline_metrics = trainer.train_and_evaluate_baselines(
        X_train, y_train, X_val, y_val
    )
    tuned_models, tuned_metrics, tuning_summary = trainer.tune_models(
        X_train, y_train, X_val, y_val
    )

    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    best_name, best_pipeline = trainer.select_best_model(all_models, all_metrics)

    final_pipeline, test_metrics, predictions_df = trainer.retrain_and_test_final_model(
        best_pipeline, X_train, y_train, X_val, y_val, X_test, y_test
    )

    model_path = trainer.save_artifacts(
        final_pipeline=final_pipeline,
        best_model_name=best_name,
        val_metrics=all_metrics,
        tuning_summary=tuning_summary,
        test_metrics=test_metrics,
        predictions_df=predictions_df,
    )

    # Verify inference interface on sample test records
    sample_preds = ModelTraining.predict_new_data(model_path, X_test.head(3))
    logger.info("Sample inference check (first 3 test students): %s", sample_preds.round(2))


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="O-Level Mathematics Score Prediction Pipeline")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    try:
        run_training_pipeline(args.config)
    except Exception as error:
        logging.critical("Pipeline execution failed: %s", error)
        sys.exit(1)
