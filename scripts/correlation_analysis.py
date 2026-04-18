import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
FILE_PATH = 'data/40343737_20260313_110600_to_112100_imu.npy'
OUTPUT_DIR = 'docs/plots/correlation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load and prepare data
data = np.load(FILE_PATH)
columns = [
    'ts', 'acc_x', 'acc_y', 'acc_z', 
    'gyr_x', 'gyr_y', 'gyr_z',
    'q_w', 'q_x', 'q_y', 'q_z'
]
df = pd.DataFrame(data, columns=columns)

# Calculate Magnitude as a derived feature for better correlation insight
df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)

# 2. Global Pearson Correlation Matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.drop(columns=['ts']).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Matriz de Correlación de Pearson (Sensores IMU)')
plt.savefig(f'{OUTPUT_DIR}/pearson_matrix.png')
print(f"Global correlation matrix saved.")

# 3. Rolling Correlation (Dynamic relationship)
# We analyze the relationship between Accel Magnitude and Gyro Magnitude over time
window_size = 100 # ~1 second assuming 100Hz
rolling_corr = df['acc_mag'].rolling(window=window_size).corr(df['gyr_mag'])

plt.figure(figsize=(12, 6))
plt.plot(rolling_corr, color='purple', label='Corr(Acc_Mag, Gyr_Mag)')
plt.axhline(y=rolling_corr.mean(), color='r', linestyle='--', label='Media')
plt.title(f'Correlación Dinámica (Ventana: {window_size} muestras)')
plt.xlabel('Muestras')
plt.ylabel('Coeficiente de Correlación')
plt.legend()
plt.grid(True)
plt.savefig(f'{OUTPUT_DIR}/rolling_correlation.png')
print(f"Rolling correlation plot saved.")

# 4. Cross-Correlation (Lags)
# Check if Gyro reacts before or after Accelerometer
def plot_cross_correlation(x, y, max_lag=100, title="Cross-Correlation"):
    lags = np.arange(-max_lag, max_lag + 1)
    cors = [x.corr(y.shift(lag)) for lag in lags]
    
    plt.figure(figsize=(10, 5))
    plt.stem(lags, cors)
    plt.title(title)
    plt.xlabel('Lag (muestras)')
    plt.ylabel('Correlación')
    plt.grid(True)
    plt.savefig(f'{OUTPUT_DIR}/cross_correlation_acc_gyr.png')

# Normalize for cross-corr
acc_norm = (df['acc_mag'] - df['acc_mag'].mean()) / df['acc_mag'].std()
gyr_norm = (df['gyr_mag'] - df['gyr_mag'].mean()) / df['gyr_mag'].std()
plot_cross_correlation(acc_norm, gyr_norm, title="Cross-Correlation: Accel Mag vs Gyro Mag")
print(f"Cross-correlation analysis completed.")

# 5. Export summary for documentation
summary = {
    "highest_correlations": corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()[1:6].to_dict(),
    "lowest_correlations": corr_matrix.unstack().sort_values(ascending=True).drop_duplicates()[:5].to_dict()
}

print("\n--- Principales Correlaciones ---")
for k, v in summary['highest_correlations'].items():
    print(f"{k}: {v:.4f}")
