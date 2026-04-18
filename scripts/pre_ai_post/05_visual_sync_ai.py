import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

# --- CONFIGURACIÓN ---
VIDEO_PATH = 'data/40343737_20260313_110600_to_112100_left.mp4'
PREDICTIONS_PATH = 'data/activity_predictions.csv'
OUTPUT_PATH = 'output/ai_sync_video.mp4'
MAX_FRAMES = 900  # 1 minuto a 15 fps para una demostración sólida
os.makedirs('output', exist_ok=True)

# 1. CARGA DE DATOS
print("Cargando predicciones de la IA...")
df_pred = pd.read_csv(PREDICTIONS_PATH)

# Definir colores para las actividades (BGR para OpenCV)
color_map = {
    'Carga': (0, 0, 255),      # Rojo
    'Descarga': (0, 255, 0),   # Verde
    'Movimiento': (0, 165, 255), # Naranja
    'Reposo': (100, 100, 100), # Gris
    'Otro': (255, 0, 0),       # Azul
    'Sin Etiqueta': (200, 200, 200)
}

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Preparar VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width + 480, height))

# 2. CONFIGURAR MATPLOTLIB PARA GRÁFICA DE IA
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(4.8, height/100), dpi=100)
canvas = FigureCanvasAgg(fig)

print(f"Generando video de sincronización IA en {OUTPUT_PATH}...")

for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret: break
    
    # Encontrar la predicción de la IA para este frame
    current_time = i / fps
    closest_idx = (df_pred['time_sec'] - current_time).abs().idxmin()
    activity = df_pred.iloc[closest_idx]['smooth_pred']
    color = color_map.get(activity, (255, 255, 255))
    
    # --- DIBUJAR EN EL FRAME ---
    # Rectángulo de fondo para el texto
    cv2.rectangle(frame, (20, 20), (600, 100), (0,0,0), -1)
    # Texto de Actividad
    cv2.putText(frame, f"IA STATUS: {str(activity).upper()}", (40, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # --- DIBUJAR GRÁFICA LATERAL ---
    ax.clear()
    # Graficar los últimos 10 segundos de aceleración
    window_data = df_pred[df_pred['time_sec'] <= current_time].tail(50)
    if not window_data.empty:
        ax.plot(window_data['time_sec'], window_data['acc_max'], color='cyan', label='Fuerza Vertical')
    ax.set_title('Sensores en Tiempo Real')
    ax.set_ylabel('m/s²')
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    
    # Marcador de actividad en la gráfica
    ax.axvspan(current_time - 0.5, current_time + 0.5, color='white', alpha=0.2)
    
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
        print(f"Sincronizando frame {i}/{MAX_FRAMES}...")

cap.release()
out.release()
plt.close(fig)
print(f"\n¡Video de sincronización IA generado! Revisa {OUTPUT_PATH}")
