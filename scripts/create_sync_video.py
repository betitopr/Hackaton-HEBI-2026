import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

# Configuración
VIDEO_PATH = 'data/40343737_20260313_110600_to_112100_left.mp4'
IMU_PATH = 'data/40343737_20260313_110600_to_112100_imu.npy'
OUTPUT_PATH = 'docs/plots/sync_check.mp4'
MAX_FRAMES = 450  # 30 segundos a 15 fps para una prueba rápida

# 1. Cargar datos
data = np.load(IMU_PATH)
acc_mag = np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2)
gyr_mag = np.sqrt(data[:, 4]**2 + data[:, 5]**2 + data[:, 6]**2)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Preparar VideoWriter (formato MP4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# El video final será el doble de ancho para poner la gráfica al lado
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width + 640, height))

# 2. Configurar Matplotlib para renderizado en memoria
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, height/100), dpi=100)
canvas = FigureCanvasAgg(fig)

print(f"Generando video de sincronización en {OUTPUT_PATH}...")

for i in range(MAX_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Limpiar y dibujar gráficas
    ax1.clear()
    ax1.plot(acc_mag[:i+1], color='cyan')
    ax1.set_title('Acelerómetro (Magnitud)')
    ax1.set_xlim(max(0, i-100), i+5) # Ventana deslizante
    ax1.grid(True, alpha=0.3)
    
    ax2.clear()
    ax2.plot(gyr_mag[:i+1], color='yellow')
    ax2.set_title('Giroscopio (Magnitud)')
    ax2.set_xlim(max(0, i-100), i+5)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    # Convertir gráfica de matplotlib a imagen OpenCV
    canvas.draw()
    rgba_buffer = canvas.buffer_rgba()
    plot_img = np.array(rgba_buffer)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    
    # Redimensionar plot_img si es necesario para que coincida con la altura del video
    if plot_img.shape[0] != height:
        plot_img = cv2.resize(plot_img, (640, height))
        
    # Combinar frame y plot
    combined = np.hstack((frame, plot_img))
    out.write(combined)
    
    if i % 50 == 0:
        print(f"Procesado frame {i}/{MAX_FRAMES}...")

cap.release()
out.release()
plt.close(fig)
print(f"\n¡Video generado exitosamente! Revisa {OUTPUT_PATH}")
