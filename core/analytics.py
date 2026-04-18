import pandas as pd
import numpy as np


class CycleAnalyzer:
    def __init__(self, df_predictions):
        self.df = df_predictions
        self.cycles = []

    def extract_metrics(self):
        """
        Analiza la secuencia de predicciones para identificar ciclos completos:
        Carga -> Movimiento -> Descarga
        """
        current_cycle = {"start_time": None, "phases": []}
        last_state = None

        # Agrupar por cambios de estado para tener una secuencia temporal
        sequence = []
        for _, row in self.df.iterrows():
            state = row["activity"]
            t = row["time_sec"]
            if state != last_state:
                sequence.append({"state": state, "time": t})
                last_state = state

        # Máquina de estados para detectar ciclos y sus tiempos
        temp_cycle = {}
        for i in range(len(sequence)):
            s = sequence[i]

            if s["state"] == "Carga":
                # Si empezamos una nueva carga, cerramos cualquier intento previo incompleto
                temp_cycle = {"carga_start": s["time"]}

            elif s["state"] == "Movimiento" and "carga_start" in temp_cycle:
                temp_cycle["movimiento_start"] = s["time"]

            elif s["state"] == "Descarga" and "movimiento_start" in temp_cycle:
                temp_cycle["descarga_start"] = s["time"]
                # Buscamos el final de la descarga (siguiente cambio de estado)
                if i + 1 < len(sequence):
                    temp_cycle["end_time"] = sequence[i + 1]["time"]
                else:
                    temp_cycle["end_time"] = s["time"] + 2.0  # Fallback

                # Calcular métricas del ciclo
                cycle_data = {
                    "total_duration": temp_cycle["end_time"]
                    - temp_cycle["carga_start"],
                    "loading_time": temp_cycle["movimiento_start"]
                    - temp_cycle["carga_start"],
                    "transport_time": temp_cycle["descarga_start"]
                    - temp_cycle["movimiento_start"],
                    "dumping_time": temp_cycle["end_time"]
                    - temp_cycle["descarga_start"],
                    "end_timestamp": temp_cycle["end_time"],
                }
                self.cycles.append(cycle_data)
                temp_cycle = {}  # Reset

        return self.calculate_summary()

    def calculate_summary(self):
        if not self.cycles:
            return None

        df_c = pd.DataFrame(self.cycles)

        # Calcular intervalos entre ciclos (Wait times / Travel back)
        # Es el tiempo desde que termina un ciclo (descarga) hasta que empieza el siguiente (carga)
        intervals = []
        for i in range(1, len(df_c)):
            interval = self.df[
                self.df["time_sec"] >= df_c.iloc[i - 1]["end_timestamp"]
            ]["time_sec"].iloc[0]
            # En realidad es más simple:
            prev_end = df_c.iloc[i - 1]["end_timestamp"]
            curr_start = df_c.iloc[i]["end_timestamp"] - df_c.iloc[i]["total_duration"]
            intervals.append(max(0, curr_start - prev_end))

        summary = {
            "total_cycles": len(df_c),
            "avg_cycle_time": df_c["total_duration"].mean(),
            "max_cycle_time": df_c["total_duration"].max(),
            "min_cycle_time": df_c["total_duration"].min(),
            "avg_interval": np.mean(intervals) if intervals else 0,
            "std_cycle_time": df_c["total_duration"].std(),  # Consistencia
            "total_productive_time": df_c["total_duration"].sum(),
        }
        return summary, df_c
