import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import re
import os

# --- CONFIGURACIÓN ---
RAW_IMU = 'data/imu_processed.csv'
RAW_LABELS = 'labels.txt'
OUTPUT_IMU = 'data/refined/imu_clean.csv'
OUTPUT_LABELS = 'data/refined/labels_clean.csv'
PLOT_DIR = 'docs/plots/cleaning'
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. FUNCIÓN: Filtrado Butterworth (Pasa-bajas)
# Para eliminar la vibración de alta frecuencia (motor) y dejar el movimiento real.
def butter_lowpass_filter(data, cutoff=1.5, fs=10.45, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 2. CARGA Y PROCESAMIENTO DE IMU
print("Leyendo IMU y aplicando filtros de señal...")
df = pd.read_csv(RAW_IMU)

# Separar componentes Vertical (Z) y Horizontal (XY) en el marco del mundo
# Esto es vital para diferenciar Excavación (Z) de Giro (XY)
df['acc_horiz'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2)
df['acc_vert'] = np.abs(df['linear_acc_z'])

# Aplicar filtrado a las señales clave
df['acc_horiz_clean'] = butter_lowpass_filter(df['acc_horiz'])
df['acc_vert_clean'] = butter_lowpass_filter(df['acc_vert'])
df['gyr_mag_clean'] = butter_lowpass_filter(np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2))

# 3. LIMPIEZA DE ETIQUETAS HUMANAS (Con buffer de seguridad adaptativo)
print("Limpiando etiquetas humanas (Margen de confianza adaptativo)...")
def parse_labels_with_margin(file_path, margin=0.3): # Reducimos margen a 0.3s para no perder descargas cortas
    labels = []
    current_min = 0
    last_sec = -1
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Corregir typo común antes de procesar
            line = line.replace("Decsarga", "Descarga")
            line = line.replace("Moviiento", "Movimiento")
            
            match_range = re.match(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*->\s*(.*)', line)
            if match_range:
                s_min, s_sec, e_min, e_sec, act = match_range.groups()
                start = int(s_min) * 60 + int(s_sec)
                end = int(e_min) * 60 + int(e_sec)
                # Aplicar margen si el segmento es largo, si es corto (<1s) usar el segmento completo
                if end - start > 0.8:
                    labels.append({'start': start + margin, 'end': end - margin, 'activity': act.split('#')[0].strip()})
                else:
                    labels.append({'start': start, 'end': end, 'activity': act.split('#')[0].strip()})
                current_min = int(e_min); last_sec = int(e_sec)
                continue
            match_sec = re.match(r'(\d+)\s*->\s*(.*)', line)
            if match_sec:
                sec = int(match_sec.group(1))
                if sec <= last_sec: current_min += 1
                start = current_min * 60 + sec
                if labels:
                    # Cerrar la actividad anterior
                    if start - labels[-1]['start'] > 0.8:
                        labels[-1]['end'] = start - margin
                    else:
                        labels[-1]['end'] = start
                labels.append({'start': start, 'end': start + 1.5, 'activity': match_sec.group(2).split('#')[0].strip()})
                last_sec = sec
    return pd.DataFrame(labels)

labels_df = parse_labels_with_margin(RAW_LABELS)

# 4. GRÁFICO DE VALIDACIÓN: Crudo vs Filtrado
print("Generando gráfico de validación de limpieza...")
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(df['time_sec'][:1000], df['acc_horiz'][:1000], color='gray', alpha=0.3, label='Cruda (Ruido Motor)')
plt.plot(df['time_sec'][:1000], df['acc_horiz_clean'][:1000], color='red', label='Limpia (Movimiento Real)')
plt.title('Limpieza de Señal: Aceleración Horizontal (Primeros 100s)')
plt.ylabel('m/s²')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(df['time_sec'][:1000], df['acc_vert'][:1000], color='gray', alpha=0.3, label='Cruda (Ruido Motor)')
plt.plot(df['time_sec'][:1000], df['acc_vert_clean'][:1000], color='blue', label='Limpia (Movimiento Real)')
plt.title('Limpieza de Señal: Aceleración Vertical (Excavación)')
plt.ylabel('m/s²')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/signal_cleaning_validation.png')

# 5. GUARDAR RESULTADOS
df.to_csv(OUTPUT_IMU, index=False)
labels_df.to_csv(OUTPUT_LABELS, index=False)
print(f"Archivos refinados guardados en data/refined/")
print(f"Gráfico de limpieza guardado en {PLOT_DIR}/signal_cleaning_validation.png")
