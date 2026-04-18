import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os

# Cargar datos procesados actuales
df = pd.read_csv('data/imu_processed.csv')

# 1. Calcular Ángulos de Euler
quats = df[['q_x', 'q_y', 'q_z', 'q_w']].values
rot = R.from_quat(quats)
euler = rot.as_euler('xyz', degrees=True)
df['roll'] = euler[:, 0]
df['pitch'] = euler[:, 1]
df['yaw'] = euler[:, 2]

# 2. Calcular Magnitud del Giroscopio
df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)

# 3. Calcular Jerk (Diferencia de aceleración lineal)
# El Jerk es la derivada de la aceleración, muy útil para detectar sacudidas de carga
df['jerk'] = df['linear_mag'].diff().fillna(0) / df['time_sec'].diff().fillna(1e-3)

# 4. Guardar dataset enriquecido
df.to_csv('data/imu_enriched.csv', index=False)
print("Dataset enriquecido guardado en data/imu_enriched.csv")
print(f"Nuevas columnas: roll, pitch, yaw, gyr_mag, jerk")
