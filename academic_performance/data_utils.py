import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

PATH = '.cache/kagglehub/datasets/jayaantanaath/student_habits_performance.csv'

def load_file(path=PATH):
    return pd.read_csv(path)

class PipelineFactory:
    def __init__(self, num_columns, ordinal_columns, nominal_columns):
        self.num_columns = num_columns
        self.ordinal_columns = ordinal_columns
        self.nominal_columns = nominal_columns

        # For numerical attributes; Scaling to values between 0 and 1, median imputing
        self.num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("std_scaler", StandardScaler())
        ])

        # For ordinal columns, add unknown parameter if not present
        self.ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("ordinal", OrdinalEncoder())
        ])

        self.nominal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("nominal", OneHotEncoder())
        ])

        self.pipeline = ColumnTransformer([
            ("numerical", self.num_pipeline, num_columns),
            ("ordinal", self.ordinal_pipeline, ordinal_columns),
            ("nominal", self.nominal_pipeline, nominal_columns)
        ])

    def fit_transform(self, data):
        return self.pipeline.fit_transform(data)

    def transform(self, data):
        return self.pipeline.transform(data)