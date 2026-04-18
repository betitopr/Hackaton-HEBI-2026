import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import os

# Configuration
FILE_PATH = 'data/40343737_20260313_110600_to_112100_imu.npy'
OUTPUT_DIR = 'docs/plots/attitude'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load data
data = np.load(FILE_PATH)
columns = [
    'ts', 'acc_x', 'acc_y', 'acc_z', 
    'gyr_x', 'gyr_y', 'gyr_z',
    'q_w', 'q_x', 'q_y', 'q_z'
]
df = pd.DataFrame(data, columns=columns)

# 2. Convert Quaternions to Euler Angles
# Note: SciPy uses (x, y, z, w) format, our data is (w, x, y, z)
quats = df[['q_x', 'q_y', 'q_z', 'q_w']].values
rot = R.from_quat(quats)
euler = rot.as_euler('xyz', degrees=True)

df['roll'] = euler[:, 0]
df['pitch'] = euler[:, 1]
df['yaw'] = euler[:, 2]

# 3. Calculate Relative Time
df['time_sec'] = (df['ts'] - df['ts'].iloc[0]) / 1e9

# 4. Plot Euler Angles
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(df['time_sec'], df['roll'], color='red', label='Roll (X)')
plt.ylabel('Grados')
plt.title('Orientación: Ángulos de Euler')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(df['time_sec'], df['pitch'], color='green', label='Pitch (Y)')
plt.ylabel('Grados')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(df['time_sec'], df['yaw'], color='blue', label='Yaw (Z)')
plt.ylabel('Grados')
plt.xlabel('Tiempo (s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/euler_angles.png')
print(f"Euler angles plot saved to {OUTPUT_DIR}/euler_angles.png")

# 5. Summary Statistics for Attitude
print("\n--- Resumen de Actitud (Grados) ---")
print(df[['roll', 'pitch', 'yaw']].describe())
