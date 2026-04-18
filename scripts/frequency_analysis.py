import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# Configuración
PROCESSED_DATA = 'data/imu_processed.csv'
OUTPUT_DIR = 'docs/plots/frequency'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Cargar datos
df = pd.read_csv(PROCESSED_DATA)
signal = df['linear_mag'].values
n = len(signal)

# Asumimos frecuencia de muestreo basada en el tiempo total
duration = df['time_sec'].iloc[-1]
fs = n / duration  # Frecuencia de muestreo (Hz)
print(f"Frecuencia de muestreo estimada: {fs:.2f} Hz")

# 2. Calcular FFT
# Restamos la media para evitar el pico masivo en 0Hz (componente DC)
yf = fft(signal - np.mean(signal))
xf = fftfreq(n, 1/fs)

# Solo nos interesa la mitad positiva del espectro
xf_pos = xf[:n//2]
yf_pos = np.abs(yf[:n//2])

# 3. Encontrar frecuencia dominante
idx_max = np.argmax(yf_pos[1:]) + 1 # Ignoramos el primer punto (cerca de 0)
dominant_freq = xf_pos[idx_max]
print(f"Frecuencia dominante encontrada: {dominant_freq:.2f} Hz")

# 4. Visualización
plt.figure(figsize=(12, 6))

# Espectro completo
plt.subplot(1, 2, 1)
plt.plot(xf_pos, yf_pos, color='blue')
plt.title('Espectro de Frecuencia (FFT) - Completo')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.grid(True, alpha=0.3)

# Zoom en bajas frecuencias (movimiento humano/vehicular)
plt.subplot(1, 2, 2)
plt.plot(xf_pos, yf_pos, color='red')
plt.xlim(0, 5)  # Zoom hasta 5Hz
plt.axvline(x=dominant_freq, color='green', linestyle='--', label=f'Dominante: {dominant_freq:.2f}Hz')
plt.title('Zoom en Bajas Frecuencias (0-5 Hz)')
plt.xlabel('Frecuencia (Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fft_analysis.png')
print(f"Análisis FFT guardado en {OUTPUT_DIR}/fft_analysis.png")

# 5. Guardar resumen para docs
with open('docs/frequency_summary.txt', 'w') as f:
    f.write(f"ANÁLISIS DE FRECUENCIA\n")
    f.write(f"Frecuencia de Muestreo: {fs:.2f} Hz\n")
    f.write(f"Frecuencia Dominante: {dominant_freq:.2f} Hz\n")
    if 0.5 < dominant_freq < 3.0:
        f.write("Interpretación: La frecuencia dominante está en el rango típico de la marcha humana o vibración vehicular suave.\n")
