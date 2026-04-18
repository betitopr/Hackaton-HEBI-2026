from abc import ABC, abstractmethod
import pandas as pd


class BaseExcavatorModel(ABC):
    """Interfaz estándar para cualquier modelo de IA en el proyecto."""

    @abstractmethod
    def load(self, model_path: str):
        """Carga el modelo desde disco."""
        pass

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Recibe un DataFrame de características y devuelve las etiquetas predichas."""
        pass
