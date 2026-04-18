import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuración
PROCESSED_DATA = 'data/imu_processed.csv'
OUTPUT_DIR = 'docs/plots/events'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Cargar datos procesados
df = pd.read_csv(PROCESSED_DATA)

# 2. Definir Umbrales (Heurísticos basados en el EDA previo)
# Estos pueden ajustarse según se vea el resultado
ACCEL_THRESHOLD = 3.5  # m/s^2 por encima de la calma (que es 0)
GYRO_THRESHOLD = 1.5   # rad/s (giros rápidos)
STILL_THRESHOLD = 0.5  # m/s^2 por debajo de esto se considera reposo

# 3. Detectar Eventos
# Impactos: Magnitud lineal alta
df['event_impact'] = df['linear_mag'] > ACCEL_THRESHOLD

# Giros bruscos: Magnitud de giroscopio alta
df['gyr_mag'] = np.sqrt(df['gyr_x']**2 + df['gyr_y']**2 + df['gyr_z']**2)
df['event_turn'] = df['gyr_mag'] > GYRO_THRESHOLD

# Reposo: Magnitud lineal muy baja
df['event_still'] = df['linear_mag'] < STILL_THRESHOLD

# 4. Extraer timestamps de eventos para el informe
def get_event_segments(series, time_series):
    segments = []
    start_time = None
    for i in range(len(series)):
        if series[i] and start_time is None:
            start_time = time_series[i]
        elif not series[i] and start_time is not None:
            segments.append((start_time, time_series[i-1]))
            start_time = None
    return segments

impacts = get_event_segments(df['event_impact'], df['time_sec'])
turns = get_event_segments(df['event_turn'], df['time_sec'])

print(f"--- Eventos Detectados ---")
print(f"Impactos detectados: {len(impacts)}")
print(f"Giros bruscos detectados: {len(turns)}")

# 5. Visualización del Timeline de Eventos
plt.figure(figsize=(15, 8))

# Graficar señales base
plt.plot(df['time_sec'], df['linear_mag'], color='gray', alpha=0.3, label='Acel. Lineal')
plt.plot(df['time_sec'], df['gyr_mag'], color='orange', alpha=0.3, label='Giro Mag')

# Resaltar eventos
for start, end in impacts:
    plt.axvspan(start, end, color='red', alpha=0.5, label='Impacto' if start == impacts[0][0] else "")

for start, end in turns:
    plt.axvspan(start, end, color='blue', alpha=0.3, label='Giro Brusco' if start == turns[0][0] else "")

plt.title('Timeline de Eventos Detectados')
plt.xlabel('Tiempo (s)')
plt.ylabel('Magnitud')
plt.legend(loc='upper right')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/event_timeline.png')
print(f"Timeline guardado en {OUTPUT_DIR}/event_timeline.png")

# 6. Guardar lista de eventos para el informe MD
with open('docs/detected_events.txt', 'w') as f:
    f.write("LISTA DE IMPACTOS DETECTADOS (segundos):\n")
    for s, e in impacts:
        f.write(f"- Inicio: {s:.2f}s, Fin: {e:.2f}s (Duración: {e-s:.2f}s)\n")
    f.write("\nLISTA DE GIROS BRUSCOS (segundos):\n")
    for s, e in turns:
        f.write(f"- Inicio: {s:.2f}s, Fin: {e:.2f}s (Duración: {e-s:.2f}s)\n")
