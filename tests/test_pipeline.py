import tempfile
import pandas as pd
import pytest
import yaml

from main import load_and_validate_config
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


@pytest.fixture
def sample_config():
    return {
        "file_path": "dummy.csv",
        "target_column": "final_test",
        "output_dir": "test_artifacts",
        "random_state": 42,
        "val_test_size": 0.4,
        "val_size": 0.5,
        "cv_folds": 2,
        "numerical_features": ["age", "attendance_rate", "duration_hours", "total_class_size"],
        "nominal_features": ["CCA", "learning_style"],
        "passthrough_features": ["gender", "tuition", "direct_admission"],
        "columns_to_drop": [
            "student_id", "mode_of_transport", "bag_color",
            "sleep_time", "wake_time", "n_male", "n_female", "index"
        ],
        "param_grids": {
            "ridge": {"regressor__alpha": [1.0, 10.0]},
        },
    }


def test_config_validation_missing_key():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump({"file_path": "data.csv"}, tmp)
        tmp_path = tmp.name

    with pytest.raises(ValueError, match="Missing required configuration keys"):
        load_and_validate_config(tmp_path)


def test_clean_data_edge_cases(sample_config):
    raw_df = pd.DataFrame({
        "student_id": list((101, 101, 102, 103)),
        "age": list((15, 15, -5, -4)),
        "CCA": ["ARTS", "ARTS", "None", "Sports"],
        "tuition": ["Y", "Y", "No", "N"],
        "gender": ["Male", "Male", "Female", "Female"],
        "direct_admission": ["Yes", "Yes", "No", "Yes"],
        "learning_style": ["Visual", "Visual", "Auditory", "Kinesthetic"],
        "sleep_time": ["23.30", "23.30", "01.00", "22.00"],
        "wake_time": ["06.30", "06.30", "07.30", "06.00"],
        "n_male": list((20, 20, 18, 22)),
        "n_female": list((20, 20, 22, 18)),
        "attendance_rate": [95.0, 95.0, 88.0, 90.0],
        "final_test": [78.0, 78.0, 65.0, 70.0],
    })

    prep = DataPreparation(sample_config)
    cleaned = prep.clean_data(raw_df, is_training=True)

    assert len(cleaned) == 2
    assert list(cleaned["age"]) == list((15, 15))
    assert list(cleaned["CCA"]) == ["Arts", "No_CCA"]
    assert list(cleaned["tuition"]) == list((1, 0))
    assert list(cleaned["duration_hours"]) == [7.0, 6.5]
    assert list(cleaned["total_class_size"]) == list((40, 40))


def test_end_to_end_smoke_pipeline(sample_config, tmp_path):
    sample_config["output_dir"] = str(tmp_path / "artifacts")

    df = pd.DataFrame({
        "student_id": list(range(1, 21)),
        "age": list((15, 16)) * 10,
        "CCA": ["Arts", "Sports", "Clubs", "None"] * 5,
        "tuition": ["Yes", "No"] * 10,
        "gender": ["Male", "Female"] * 10,
        "direct_admission": ["Yes", "No"] * 10,
        "learning_style": ["Visual", "Auditory"] * 10,
        "sleep_time": ["23:00", "00:30"] * 10,
        "wake_time": ["06:30", "07:00"] * 10,
        "n_male": [20] * 20,
        "n_female": [20] * 20,
        "attendance_rate": [80.0 + i for i in range(20)],
        "final_test": [55.0 + i * 1.5 for i in range(20)],
    })

    prep = DataPreparation(sample_config)
    cleaned = prep.clean_data(df, is_training=True)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split_data(cleaned)

    trainer = ModelTraining(sample_config, prep.preprocessor)
    base_models, base_metrics = trainer.train_and_evaluate_baselines(X_train, y_train, X_val, y_val)
    tuned_models, tuned_metrics, summary = trainer.tune_models(X_train, y_train, X_val, y_val)

    best_name, best_pipe = trainer.select_best_model(
        {**base_models, **tuned_models}, {**base_metrics, **tuned_metrics}
    )
    final_pipe, test_metrics, preds_df = trainer.retrain_and_test_final_model(
        best_pipe, X_train, y_train, X_val, y_val, X_test, y_test
    )
    saved_path = trainer.save_artifacts(
        final_pipe, best_name, base_metrics, summary, test_metrics, preds_df
    )

    assert "R2" in test_metrics
    assert len(ModelTraining.predict_new_data(saved_path, X_test)) == len(X_test)

