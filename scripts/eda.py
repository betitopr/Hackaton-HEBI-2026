import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
file_path = 'data/40343737_20260313_110600_to_112100_imu.npy'
data = np.load(file_path)

# Columns assumption based on 11 columns: 
# 0: Timestamp, 1-3: Accel, 4-6: Gyro, 7-10: Quaternion
columns = [
    'timestamp', 
    'accel_x', 'accel_y', 'accel_z', 
    'gyro_x', 'gyro_y', 'gyro_z',
    'quat_w', 'quat_x', 'quat_y', 'quat_z'
]

df = pd.DataFrame(data, columns=columns)

# Basic Analysis
print("--- Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# Convert timestamp to relative time (seconds) if it looks like nanoseconds
if df['timestamp'].iloc[0] > 1e12:
    df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
else:
    df['time_sec'] = df['timestamp'] - df['timestamp'].iloc[0]

# Plotting
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Accel
axes[0].plot(df['time_sec'], df['accel_x'], label='X')
axes[0].plot(df['time_sec'], df['accel_y'], label='Y')
axes[0].plot(df['time_sec'], df['accel_z'], label='Z')
axes[0].set_title('Accelerometer (m/s²)')
axes[0].legend()
axes[0].grid(True)

# Gyro
axes[1].plot(df['time_sec'], df['gyro_x'], label='X')
axes[1].plot(df['time_sec'], df['gyro_y'], label='Y')
axes[1].plot(df['time_sec'], df['gyro_z'], label='Z')
axes[1].set_title('Gyroscope (rad/s)')
axes[1].legend()
axes[1].grid(True)

# Quaternions
axes[2].plot(df['time_sec'], df['quat_w'], label='W')
axes[2].plot(df['time_sec'], df['quat_x'], label='X')
axes[2].plot(df['time_sec'], df['quat_y'], label='Y')
axes[2].plot(df['time_sec'], df['quat_z'], label='Z')
axes[2].set_title('Quaternion (Orientation)')
axes[2].legend()
axes[2].grid(True)
axes[2].set_xlabel('Time (s)')

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/imu_analysis.png')
print("\nPlot saved to plots/imu_analysis.png")
