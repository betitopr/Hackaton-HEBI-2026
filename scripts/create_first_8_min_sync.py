import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os
import joblib

# --- CONFIGURACIÓN ---
VIDEO_PATH = 'data/40343737_20260313_110600_to_112100_left.mp4'
PREDICTIONS_PATH = 'data/activity_predictions.csv'
OUTPUT_PATH = 'output/ai_sync_first_8min.mp4'
DURATION_LIMIT_SEC = 480 # 8 minutos
os.makedirs('output', exist_ok=True)

# 1. CARGA DE DATOS Y MODELO HÍBRIDO
print("Cargando modelo Híbrido Robusto...")
hybrid_data = joblib.load('data/refined/hybrid_model.pkl')
rf_model = hybrid_data['model']
thresholds = hybrid_data['thresholds']

df_pred = pd.read_csv(PREDICTIONS_PATH)

# Colores (BGR para OpenCV)
color_map = {
    'Carga': (0, 0, 255), 'Descarga': (0, 255, 0), 
    'Movimiento': (0, 165, 255), 'Reposo': (100, 100, 100), 'Otro': (255, 0, 0)
}

def get_hybrid_prediction(row):
    if row['gyr_mean'] > thresholds['gyro_move']: return 'Movimiento'
    if row['acc_v_std'] < thresholds['acc_reposo'] and row['acc_h_std'] < thresholds['acc_reposo']: return 'Reposo'
    cols_to_drop = ['time_sec', 'label', 'robust_label', 'target', 'smooth_pred', 'raw_pred', 'activity']
    X_row = row.drop(cols_to_drop, errors='ignore')
    
    # Convertir a DataFrame de una fila para mantener los nombres de las columnas
    X_df = pd.DataFrame([X_row])
    return rf_model.predict(X_df)[0]

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_to_process = int(DURATION_LIMIT_SEC * fps)

# Preparar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width + 480, height))

# 2. CONFIGURAR MATPLOTLIB
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(4.8, height/100), dpi=100)
canvas = FigureCanvasAgg(fig)

print(f"Generando video de los primeros 8 minutos en {OUTPUT_PATH}...")

for i in range(total_frames_to_process):
    ret, frame = cap.read()
    if not ret: break
    
    current_time = i / fps
    
    # Inferencia Híbrida
    closest_idx = (df_pred['time_sec'] - current_time).abs().idxmin()
    sensor_features = df_pred.iloc[closest_idx]
    activity = get_hybrid_prediction(sensor_features)
    color = color_map.get(activity, (255, 255, 255))
    
    # --- UI EN VIDEO ---
    cv2.rectangle(frame, (20, 20), (550, 120), (0,0,0), -1)
    cv2.putText(frame, f"ACTIVITY: {str(activity).upper()}", (40, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    cv2.putText(frame, f"TIMESTAMP: {current_time:.2f}s / {DURATION_LIMIT_SEC}s", (40, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # --- GRÁFICA LATERAL ---
    if i % 2 == 0: # Solo actualizar la gráfica cada 2 frames para ganar velocidad
        ax.clear()
        window_data = df_pred[(df_pred['time_sec'] > current_time - 4) & (df_pred['time_sec'] <= current_time)]
        if not window_data.empty:
            y_col = 'acc_h_mean' if 'acc_h_mean' in window_data.columns else 'acc_max'
            ax.plot(window_data['time_sec'], window_data[y_col], color='cyan', linewidth=2)
            ax.fill_between(window_data['time_sec'], window_data[y_col], color='cyan', alpha=0.2)
        ax.set_title('Sensores en Tiempo Real', fontsize=15)
        ax.set_ylim(0, 12)
        ax.grid(True, alpha=0.3)
        ax.axvline(current_time, color='white', linestyle='--', alpha=0.8)
        fig.tight_layout()
        canvas.draw()
        plot_img = np.array(canvas.buffer_rgba())
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        if plot_img.shape[0] != height:
            plot_img = cv2.resize(plot_img, (480, height))
        
    combined = np.hstack((frame, plot_img))
    out.write(combined)
    
    if i % 500 == 0:
        print(f"Progreso: {current_time:.1f}s procesados...")

cap.release()
out.release()
plt.close(fig)
print(f"\n¡Video de 8 minutos generado! Ruta: {OUTPUT_PATH}")
