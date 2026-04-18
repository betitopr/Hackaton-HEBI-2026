import pandas as pd
import subprocess
import os

# Configuración
PROCESSED_DATA = 'data/imu_processed.csv'
VIDEO_FILE = 'data/40343737_20260313_110600_to_112100_left.mp4'
OUTPUT_DIR = 'docs/clips'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Cargar datos
df = pd.read_csv(PROCESSED_DATA)

# 2. Encontrar los Top 5 impactos más fuertes (basados en linear_mag)
# Evitamos duplicados cercanos para no sacar clips del mismo evento
top_impacts = df.sort_values(by='linear_mag', ascending=False)
selected_events = []
min_dist_sec = 5.0 # Mínimo 5 segundos entre clips seleccionados

for _, row in top_impacts.iterrows():
    t = row['time_sec']
    mag = row['linear_mag']
    
    # Verificar si está muy cerca de uno ya seleccionado
    if all(abs(t - prev_t) > min_dist_sec for prev_t, _ in selected_events):
        selected_events.append((t, mag))
    
    if len(selected_events) >= 5:
        break

print(f"Top 5 eventos seleccionados para segmentación:")
for i, (t, mag) in enumerate(selected_events):
    print(f"{i+1}. Tiempo: {t:.2f}s, Magnitud: {mag:.2f} m/s²")

# 3. Recortar clips con FFmpeg
# Ventana: 2 segundos antes y 2 después del pico
WINDOW = 2.0

for i, (t, mag) in enumerate(selected_events):
    start_time = max(0, t - WINDOW)
    duration = WINDOW * 2
    output_name = f"{OUTPUT_DIR}/impact_top_{i+1}_{int(t)}s.mp4"
    
    # Comando ffmpeg: rápido (sin re-encodear si es posible, o re-encodeando rápido)
    cmd = [
        'ffmpeg', '-y', '-ss', str(start_time), '-t', str(duration),
        '-i', VIDEO_FILE, '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        output_name
    ]
    
    print(f"Generando clip {i+1}...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Clip guardado: {output_name}")

print("\nSegmentación completada.")
