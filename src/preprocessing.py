from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

@dataclass
class PreprocessArtifacts:
    pipeline: Pipeline
    numeric_features: list
    categorical_features: list

def build_preprocess_pipeline(df: pd.DataFrame, target_col: str) -> PreprocessArtifacts:
    features = [c for c in df.columns if c != target_col]
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in features if c not in numeric_features]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler(with_mean=False))])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor)])
    return PreprocessArtifacts(pipeline=pipe, numeric_features=numeric_features, categorical_features=categorical_features)

def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y
