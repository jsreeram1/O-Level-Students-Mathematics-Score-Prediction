# Standard library imports
import logging
import re
from typing import Any, Dict

# Related third-party imports
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreparation:
    """
    A class used to clean and preprocess O Level Maths Score Data.

    Attributes:
    -----------
    config : Dict[str, Any]
        Configuration dictionary containing parameters for data cleaning and preprocessing.
    preprocessor : sklearn.compose.ColumnTransformer
        A preprocessor pipeline for transforming numerical and nominal features.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataPreparation class with a configuration dictionary.

        Args:
        -----
        config (Dict[str, Any]): Configuration dictionary containing parameters for data cleaning and preprocessing.
        """
        self.config = config
        self.preprocessor = self._create_preprocessor()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the input DataFrame by performing several preprocessing steps.

        Args:
        -----
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        --------
        pd.DataFrame: The cleaned DataFrame.
        """
        logging.info("Starting data cleaning.")

        df.drop_duplicates(subset= 'student_id', inplace=True)
        # Drop only rows where the final_test score, attendance_rate are missing 
        df = df.dropna(subset=['final_test', 'attendance_rate'])
        df['final_test']=pd.to_numeric(df['final_test'], errors='coerce')
        df['attendance_rate']=pd.to_numeric(df['attendance_rate'], errors='coerce')
        # Convert both 'None' and 'NONE' to nan and then replace it with 'No_CCA'
        df['CCA'] = df['CCA'].replace(['None', 'NONE'], np.nan)
        df['CCA'] = df['CCA'].fillna('No_CCA')
        # Standardize the values of the columns 'CCA', 'tuition'
        df['CCA'] = df['CCA'].replace({'ARTS': 'Arts', 'SPORTS': 'Sports', 'CLUBS': 'Clubs'})
        df['tuition'] = df['tuition'].replace({'Y': 'Yes', 'N': 'No'}) 
        # Convert binary text columns directly into 0 and 1
        df['gender'] = df['gender'].astype(str).str.lower().map({'male': 1, 'female': 0})
        df['tuition'] = df['tuition'].astype(str).str.lower().map({'Yes': 1, 'No': 0})
        df['direct_admission'] = df['direct_admission'].astype(str).str.lower().map({'Yes': 1, 'No': 0})
        # Drop rows where age is exactly -4
        df = df[df["age"] != -4]
        # Fix the typos for -5, 5, and 6
        df["age"] = df["age"].replace({-5: 15, 5: 15, 6: 16})
        # Convert the string times (like '22.30') into an official datetime format.
        # Map them to a baseline dummy date (e.g., 1900-01-01) just to do the time math.
        sleep_datetime = pd.to_datetime(
    df["sleep_time"].str.replace(".", ":", regex=False), format="%H:%M"
)
        wake_datetime = pd.to_datetime(
        df["wake_time"].str.replace(".", ":", regex=False), format="%H:%M"
        )
        # Subtract the two datetime series to get the raw time difference (Duration)
        raw_duration = wake_datetime - sleep_datetime
        # Convert that time difference into numeric hours (e.g., 7.5)
        df["duration_hours"] = raw_duration.dt.total_seconds() / 3600
        # Fix Midnight Crossovers
        # If the sleep time was 22:00 and wake was 05:30, the math results in a negative number.
        # If it's negative, we add 24 hours to get the real duration.
        df["duration_hours"] = np.where(
        df["duration_hours"] < 0, df["duration_hours"] + 24, df["duration_hours"]
        )
        # Combines n_male and n_female into a comprehensive single layout
        if "n_male" in df.columns and "n_female" in df.columns:
            df["total_class_size"] = df["n_male"] + df["n_female"]
        # Columns to drop
        columns_to_drop = ['student_id', 'mode_of_transport', 'bag_color', 'sleep_time', 'wake_time', 'n_male', 'n_female', 'index']
        df = df.drop(columns =  columns_to_drop, errors='ignore')
    
        logging.info("Data cleaning completed.")
        return df

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Creates a preprocessor pipeline for transforming numerical, and nominal features.

        Returns:
        --------
        sklearn.compose.ColumnTransformer: A ColumnTransformer object for preprocessing the data.
        """
        numerical_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        nominal_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
       
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )
        return preprocessor
     