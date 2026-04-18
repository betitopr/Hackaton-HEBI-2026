import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# 1. Función para parsear el archivo de etiquetas humano
def parse_labels(file_path):
    labels = []
    current_min = 0
    last_sec = -1
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Intentar capturar formato MM:SS o SS
            # Ejemplos: "00:00 - 00:04 -> Reposo", "27 -> Descarga", "01:04 - 01:12 -> movimiento"
            
            # Caso 1: Rango MM:SS - MM:SS
            match_range = re.match(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*->\s*(.*)', line)
            if match_range:
                s_min, s_sec, e_min, e_sec, act = match_range.groups()
                start = int(s_min) * 60 + int(s_sec)
                end = int(e_min) * 60 + int(e_sec)
                labels.append((start, end, act.split('#')[0].strip()))
                current_min = int(e_min)
                last_sec = int(e_sec)
                continue

            # Caso 2: Punto de inicio MM:SS -> Actividad
            match_start = re.match(r'(\d+):(\d+)\s*->\s*(.*)', line)
            if match_start:
                s_min, s_sec, act = match_start.groups()
                start = int(s_min) * 60 + int(s_sec)
                if labels:
                    labels[-1] = (labels[-1][0], start, labels[-1][2])
                labels.append((start, start + 5, act.split('#')[0].strip())) # End provisional
                current_min = int(s_min)
                last_sec = int(s_sec)
                continue

            # Caso 3: Solo Segundos -> Actividad (implica avance de tiempo)
            match_sec = re.match(r'(\d+)\s*->\s*(.*)', line)
            if match_sec:
                sec = int(match_sec.group(1))
                # Si el segundo es menor al anterior, es que pasamos al siguiente minuto
                if sec <= last_sec:
                    current_min += 1
                
                start = current_min * 60 + sec
                if labels:
                    # Actualizar el fin de la actividad anterior
                    prev_start, _, prev_act = labels[-1]
                    labels[-1] = (prev_start, start, prev_act)
                
                labels.append((start, start + 5, match_sec.group(2).split('#')[0].strip()))
                last_sec = sec
                continue

    return labels

# 2. Cargar datos IMU y Etiquetas
df = pd.read_csv('data/imu_processed.csv')
if 'gyr_mag' not in df.columns:
    df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
if 'linear_mag' not in df.columns:
    df['linear_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)

human_labels = parse_labels('labels.txt')

# 3. Visualización de Correlación
plt.figure(figsize=(20, 10))

# Graficar Aceleración Lineal (Mag) y Giro (Mag)
plt.plot(df['time_sec'], df['linear_mag'], color='black', alpha=0.2, label='Acel. Lineal')
plt.plot(df['time_sec'], df['gyr_mag'], color='blue', alpha=0.2, label='Giro Mag')

# Colores por actividad
color_map = {
    'Carga': 'red',
    'Descarga': 'green',
    'Movimiento': 'orange',
    'Reposo': 'gray'
}

for start, end, act in human_labels:
    # Buscar color (simplificando el nombre de la actividad)
    color = 'yellow' # default
    for key in color_map:
        if key.lower() in act.lower():
            color = color_map[key]
            break
    
    plt.axvspan(start, end, color=color, alpha=0.3)
    # Poner texto arriba
    plt.text((start + end)/2, plt.ylim()[1]*0.9, act, rotation=90, fontsize=8, verticalalignment='top')

plt.title('Validación de Etiquetas Humanas vs Sensores IMU')
plt.xlabel('Tiempo (s)')
plt.ylabel('Magnitud')
plt.xlim(0, max(df['time_sec']))
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/plots/human_label_validation.png')
print("Visualización generada en docs/plots/human_label_validation.png")

# 4. Análisis de Firmas (Promedios por Actividad)
stats = []
for start, end, act in human_labels:
    mask = (df['time_sec'] >= start) & (df['time_sec'] <= end)
    segment = df[mask]
    if not segment.empty:
        stats.append({
            'act': act,
            'acc_std': segment['linear_mag'].std(),
            'gyr_mean': segment['gyr_mag'].mean(),
            'pitch_mean': np.arctan2(-segment['linear_acc_x'], np.sqrt(segment['linear_acc_y']**2 + segment['linear_acc_z']**2)).mean()
        })

stats_df = pd.DataFrame(stats)
# Agrupar por tipo de actividad (simplificado)
stats_df['category'] = stats_df['act'].apply(lambda x: next((k for k in color_map if k.lower() in x.lower()), 'Otro'))
summary = stats_df.groupby('category').mean(numeric_only=True)
print("\n--- FIRMAS DIGITALES POR ACTIVIDAD ---")
print(summary)
