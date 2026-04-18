import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# Configuración
FILE_PATH = os.environ.get(
    "IMU_NPY_INPUT", "data/40343737_20260313_110600_to_112100_imu.npy"
)
OUTPUT_DIR = "docs/plots/preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Cargar datos
data = np.load(FILE_PATH)
columns = [
    "ts",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "q_w",
    "q_x",
    "q_y",
    "q_z",
]
df = pd.DataFrame(data, columns=columns)
df["time_sec"] = (df["ts"] - df["ts"].iloc[0]) / 1e9

# 2. Extracción de Aceleración Lineal en el Marco del Mundo (World Frame)
# SciPy usa (x, y, z, w), nuestros datos están en (w, x, y, z)
quats = df[["q_x", "q_y", "q_z", "q_w"]].values.copy()
accel_body = df[["acc_x", "acc_y", "acc_z"]].values.copy()

# Rotar aceleración del cuerpo al marco del mundo
rotations = R.from_quat(quats)
accel_world = rotations.apply(accel_body)

# Definir vector de gravedad en el mundo.
# Basado en el EDA, la gravedad estaba mayormente en Y.
# Al rotar al mundo, la gravedad debería estar alineada con un eje constante.
# Estimamos el vector de gravedad promedio en el marco del mundo para restarlo.
gravity_world = np.mean(
    accel_world[:100], axis=0
)  # Asumimos reposo inicial para calibrar
print(f"Vector de gravedad estimado en el mundo: {gravity_world}")

accel_linear = accel_world - gravity_world

# 3. Guardar en el DataFrame
df["linear_acc_x"] = accel_linear[:, 0]
df["linear_acc_y"] = accel_linear[:, 1]
df["linear_acc_z"] = accel_linear[:, 2]

# Calcular Magnitud de Movimiento Puro
df["linear_mag"] = np.sqrt(
    df["linear_acc_x"] ** 2 + df["linear_acc_y"] ** 2 + df["linear_acc_z"] ** 2
)

# 4. Visualización Comparativa
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(
    df["time_sec"], df["acc_y"], color="gray", alpha=0.5, label="Original (Body Y)"
)
plt.title("Aceleración Cruda vs Lineal (Mundo)")
plt.ylabel("m/s²")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df["time_sec"], df["linear_acc_x"], label="Linear X (World)", color="red")
plt.plot(df["time_sec"], df["linear_acc_y"], label="Linear Y (World)", color="green")
plt.plot(df["time_sec"], df["linear_acc_z"], label="Linear Z (World)", color="blue")
plt.ylabel("m/s²")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(
    df["time_sec"], df["linear_mag"], color="black", label="Magnitud Movimiento Puro"
)
plt.ylabel("m/s²")
plt.xlabel("Tiempo (s)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/linear_acceleration.png")
print(f"Gráfico de aceleración lineal guardado en {OUTPUT_DIR}/linear_acceleration.png")

# 5. Guardar datos procesados para futuros pasos de IA
processed_file = "data/imu_processed.csv"
df.to_csv(processed_file, index=False)
print(f"Datos pre-procesados guardados en {processed_file}")
