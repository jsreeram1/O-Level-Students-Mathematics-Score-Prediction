import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


class DataPreparation:
    """Handles data loading, validation, cleaning, feature engineering, and splitting."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self) -> ColumnTransformer:
        """Creates the sklearn ColumnTransformer for numerical, nominal, and binary features."""
        num_features = self.config.get("numerical_features", [])
        nom_features = self.config.get("nominal_features", [])
        pass_features = self.config.get("passthrough_features", [])

        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        nominal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        return ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, num_features),
                ("nom", nominal_transformer, nom_features),
                ("pass", "passthrough", pass_features),
            ],
            remainder="drop",
            n_jobs=-1,
        )

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads CSV data and validates that required schema columns exist."""
        try:
            df = pd.read_csv(file_path)
            logger.info("Loaded dataset from %s with shape %s", file_path, df.shape)
        except FileNotFoundError as exc:
            logger.error("Dataset file not found at %s", file_path)
            raise exc
        except Exception as exc:
            logger.error("Failed to read CSV file: %s", exc)
            raise exc

        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        target_col = self.config["target_column"]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' is missing from dataset.")

        return df

    def clean_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Runs modular data cleaning and feature engineering steps."""
        logger.info("Starting data cleaning (initial shape: %s)", df.shape)
        df = df.copy()

        df = self._remove_duplicates_and_nulls(df, is_training=is_training)
        df = self._fix_ages(df)
        df = self._standardize_categories(df)
        df = self._encode_binary_columns(df)
        df = self._compute_sleep_duration(df)
        df = self._compute_class_size(df)
        df = self._drop_unused_columns(df)

        logger.info("Completed data cleaning (final shape: %s)", df.shape)
        return df

    def _remove_duplicates_and_nulls(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        if "student_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["student_id"], keep="first")
            logger.info("Dropped %d duplicate student_id rows", before - len(df))

        if is_training:
            target_col = self.config["target_column"]
            cols_to_check = [c for c in [target_col, "attendance_rate"] if c in df.columns]
            before = len(df)
            df = df.dropna(subset=cols_to_check)
            logger.info("Dropped %d rows with missing %s", before - len(df), cols_to_check)

        return df

    def _fix_ages(self, df: pd.DataFrame) -> pd.DataFrame:
        if "age" not in df.columns:
            return df

        before = len(df)
        df = df[df["age"] != -4].copy()
        logger.info("Dropped %d rows with invalid age -4", before - len(df))

        df["age"] = df["age"].replace({-5: 15, 5: 15, 6: 16})
        return df

    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if "CCA" in df.columns:
            df["CCA"] = (
                df["CCA"]
                .fillna("No_CCA")
                .astype(str)
                .str.strip()
                .replace({
                    "None": "No_CCA",
                    "NONE": "No_CCA",
                    "nan": "No_CCA",
                    "ARTS": "Arts",
                    "SPORTS": "Sports",
                    "CLUBS": "Clubs",
                })
            )

        if "tuition" in df.columns:
            df["tuition"] = (
                df["tuition"]
                .astype(str)
                .str.strip()
                .replace({"Y": "Yes", "N": "No"})
            )

        return df

    def _encode_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if "gender" in df.columns:
            df["gender"] = df["gender"].astype(str).str.strip().str.lower().map({"male": 1, "female": 0})

        if "tuition" in df.columns:
            df["tuition"] = df["tuition"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

        if "direct_admission" in df.columns:
            df["direct_admission"] = (
                df["direct_admission"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
            )

        return df

    def _compute_sleep_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sleep_time" in df.columns and "wake_time" in df.columns:
            try:
                sleep_dt = pd.to_datetime(
                    df["sleep_time"].astype(str).str.replace(".", ":", regex=False), format="%H:%M"
                )
                wake_dt = pd.to_datetime(
                    df["wake_time"].astype(str).str.replace(".", ":", regex=False), format="%H:%M"
                )
                raw_hours = (wake_dt - sleep_dt).dt.total_seconds() / 3600.0
                df["duration_hours"] = np.where(raw_hours < 0, raw_hours + 24.0, raw_hours)
            except Exception as exc:
                logger.error("Error parsing sleep_time/wake_time: %s", exc)
                raise exc

        return df

    def _compute_class_size(self, df: pd.DataFrame) -> pd.DataFrame:
        if "n_male" in df.columns and "n_female" in df.columns:
            df["total_class_size"] = df["n_male"] + df["n_female"]
        return df

    def _drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = self.config.get("columns_to_drop", [])
        return df.drop(columns=drop_cols, errors="ignore")

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Splits cleaned dataset into train, validation, and test sets using configured seed."""
        target_col = self.config["target_column"]
        seed = self.config["random_state"]
        val_test_size = self.config["val_test_size"]
        val_size = self.config["val_size"]

        if not (0 < val_test_size < 1) or not (0 < val_size < 1):
            raise ValueError("val_test_size and val_size must be strictly between 0 and 1.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=val_test_size, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=seed
        )

        logger.info(
            "Data split complete -> Train: %d, Val: %d, Test: %d",
            len(X_train), len(X_val), len(X_test)
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
