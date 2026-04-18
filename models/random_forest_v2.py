import joblib
import pandas as pd
from core.base_model import BaseExcavatorModel


class RandomForestExcavatorModel(BaseExcavatorModel):
    def __init__(self):
        self.model = None

    def load(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        # El modelo espera exactamente estas columnas en este orden
        # Basado en scripts/pre_ai_post/04_inference_report.py
        expected_cols = [
            "acc_h_mean",
            "acc_h_std",
            "acc_h_max",
            "acc_v_mean",
            "acc_v_std",
            "acc_v_max",
            "gyr_mean",
            "gyr_std",
            "pitch",
            "v_h_ratio",
        ]
        X = features[expected_cols]
        return pd.Series(self.model.predict(X))
