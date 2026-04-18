import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import random
import joblib

# --- CONFIGURACIÓN ---
VIDEO_PATH = 'data/40343737_20260313_110600_to_112100_left.mp4'
PREDICTIONS_PATH = 'data/activity_predictions.csv'
OUTPUT_DIR = 'output/clips'
CLIP_DURATION = 10 # segundos
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. CARGA DE DATOS Y MODELO HÍBRIDO
print("Cargando modelo Híbrido Robusto...")
hybrid_data = joblib.load('data/refined/hybrid_model.pkl')
rf_model = hybrid_data['model']
thresholds = hybrid_data['thresholds']

df_pred = pd.read_csv(PREDICTIONS_PATH) # Usado para la gráfica lateral

# Definir colores para las actividades (BGR para OpenCV)
color_map = {
    'Carga': (0, 0, 255),      # Rojo
    'Descarga': (0, 255, 0),   # Verde
    'Movimiento': (0, 165, 255), # Naranja
    'Reposo': (100, 100, 100), # Gris
    'Otro': (255, 0, 0)
}
# Lógica de predicción híbrida (Igual que en el entrenamiento)
def get_hybrid_prediction(row):
    if row['gyr_mean'] > thresholds['gyro_move']: return 'Movimiento'
    if row['acc_v_std'] < thresholds['acc_reposo'] and row['acc_h_std'] < thresholds['acc_reposo']: return 'Reposo'

    # IA solo para casos ambiguos
    # Eliminamos TODAS las columnas que puedan ser strings
    cols_to_drop = ['time_sec', 'label', 'robust_label', 'target', 'smooth_pred', 'raw_pred', 'activity']
    X_row = row.drop(cols_to_drop, errors='ignore')

    # Convertir a DataFrame de una fila para mantener los nombres de las columnas
    X_df = pd.DataFrame([X_row])
    return rf_model.predict(X_df)[0]


cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_seconds = total_frames / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 2. SELECCIÓN ALEATORIA DE TIEMPO
start_sec = random.uniform(0, max(0, total_seconds - CLIP_DURATION))
end_sec = start_sec + CLIP_DURATION
start_frame = int(start_sec * fps)
end_frame = int(end_sec * fps)

output_path = os.path.join(OUTPUT_DIR, f'ai_sync_random_{int(start_sec)}s.mp4')
print(f"Generando clip aleatorio desde {start_sec:.2f}s hasta {end_sec:.2f}s...")
print(f"Archivo de salida: {output_path}")

# Preparar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width + 480, height))

# 3. CONFIGURAR MATPLOTLIB PARA GRÁFICA DE IA
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(4.8, height/100), dpi=100)
canvas = FigureCanvasAgg(fig)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

i = start_frame
while i < end_frame:
    ret, frame = cap.read()
    if not ret: break
    
    current_time = i / fps
    
    # Encontrar las características del sensor para este frame
    closest_idx = (df_pred['time_sec'] - current_time).abs().idxmin()
    sensor_features = df_pred.iloc[closest_idx]
    
    # NUEVA PREDICCIÓN EN TIEMPO REAL USANDO EL MODELO HÍBRIDO
    activity = get_hybrid_prediction(sensor_features)
    color = color_map.get(activity, (255, 255, 255))
    
    # --- DIBUJAR EN EL FRAME ---
    cv2.rectangle(frame, (20, 20), (550, 120), (0,0,0), -1)
    cv2.putText(frame, f"ACTIVITY: {str(activity).upper()}", (40, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, f"TIMESTAMP: {current_time:.2f}s", (40, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # --- DIBUJAR GRÁFICA LATERAL ---
    ax.clear()
    # Graficar ventana de 5 segundos alrededor del tiempo actual
    window_data = df_pred[(df_pred['time_sec'] > current_time - 4) & (df_pred['time_sec'] <= current_time)]
    
    if not window_data.empty:
        # Intentamos graficar acc_h_mean si existe (del nuevo modelo unificado)
        y_col = 'acc_h_mean' if 'acc_h_mean' in window_data.columns else 'acc_max'
        ax.plot(window_data['time_sec'], window_data[y_col], color='cyan', linewidth=2)
        ax.fill_between(window_data['time_sec'], window_data[y_col], color='cyan', alpha=0.2)
    
    ax.set_title('Sensores en Tiempo Real', fontsize=15)
    ax.set_ylabel('Magnitud', fontsize=12)
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3)
    
    # Línea de tiempo actual
    ax.axvline(current_time, color='white', linestyle='--', alpha=0.8)
    
    fig.tight_layout()
    canvas.draw()
    plot_img = np.array(canvas.buffer_rgba())
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    
    if plot_img.shape[0] != height:
        plot_img = cv2.resize(plot_img, (480, height))
        
    # Combinar
    combined = np.hstack((frame, plot_img))
    out.write(combined)
    
    if i % 100 == 0:
        print(f"Procesando frame {i}/{end_frame}...")
    i += 1

cap.release()
out.release()
plt.close(fig)
print(f"\n¡Clip de sincronización IA generado con éxito!")
print(f"Ruta: {output_path}")
