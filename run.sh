#!/bin/bash

# JEBI - Pipeline de Inferencia (Solo Predicción con .npy)
# Uso: ./run.sh [ruta_al_imu.npy] [ruta_al_video.mp4]

PYTHON_VENV="/home/techatlasdev/Proyectos/hackaton/jebi/.venv/bin/python3"

IMU_INPUT=${1:-"/home/techatlasdev/Proyectos/hackaton/jebi/data/40343737_20260313_110600_to_112100_imu.npy"}
VIDEO_INPUT=${2:-"/home/techatlasdev/Proyectos/hackaton/jebi/data/40343737_20260313_110600_to_112100_left.mp4"}
MODEL_PATH="/home/techatlasdev/Proyectos/hackaton/jebi/data/refined/excavator_model.pkl"

echo "------------------------------------------------"
echo "🔍 Iniciando Inferencia JEBI (Carga desde .npy)"
echo "------------------------------------------------"

# 1. Preparar entorno y limpiar ejecuciones previas
rm -rf output_run
mkdir -p data/refined
mkdir -p output_run
mkdir -p docs/plots/inference
mkdir -p docs/plots/preprocessing
mkdir -p docs/plots/cleaning

# Verificar Ejecutable de Python
if [ ! -f "$PYTHON_VENV" ]; then
    echo "⚠️  Error: No se encuentra el entorno virtual en .venv/"
    exit 1
fi

# Verificar Modelo
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Error: No se encuentra el modelo en $MODEL_PATH"
    exit 1
fi

echo "Step 0: Extrayendo Aceleración Lineal desde .npy..."
export IMU_NPY_INPUT="$IMU_INPUT"
$PYTHON_VENV scripts/linear_accel_extraction.py

echo "Step 1: Refinando señal (Filtro Butterworth)..."
$PYTHON_VENV scripts/pre_ai_post/01_clean_and_refine.py

echo "Step 2: Ejecutando Inferencia y KPIs..."
$PYTHON_VENV run_pipeline.py

echo "Step 3: Generando Gráficas de Inferencia..."
$PYTHON_VENV -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

if not os.path.exists('output/final_predictions_v3.csv'):
    print('Error: No se generaron predicciones.')
    exit(1)

df = pd.read_csv('output/final_predictions_v3.csv')
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='activity', palette='viridis', hue='activity', legend=False)
plt.title('Distribución de Actividades Detectadas')
plt.savefig('output_run/activity_distribution.png')

plt.figure(figsize=(12, 4))
plt.plot(df['time_sec'], df['acc_v_max'], alpha=0.5, label='Esfuerzo Vertical')
plt.scatter(df['time_sec'], [5]*len(df), c=df['activity'].astype('category').cat.codes, cmap='tab10', s=2, label='Actividad IA')
plt.title('Timeline de Actividades vs Telemetría')
plt.legend()
plt.savefig('output_run/inference_timeline.png')
"

echo "Step 4: Generando Video Sincronizado (IA)..."
export INPUT_VIDEO_SYNC="$VIDEO_INPUT"
export OUTPUT_VIDEO_SYNC="output_run/jebi_inference_sync.mp4"

$PYTHON_VENV -c "
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

VIDEO_PATH = os.environ['INPUT_VIDEO_SYNC']
PREDICTIONS_PATH = 'output/final_predictions_v3.csv'
OUTPUT_PATH = os.environ['OUTPUT_VIDEO_SYNC']

if not os.path.exists(VIDEO_PATH):
    print(f'Error: Video no encontrado en {VIDEO_PATH}')
    exit(1)

df_pred = pd.read_csv(PREDICTIONS_PATH)
color_map = {'Carga': (0,0,255), 'Descarga': (0,255,0), 'Movimiento': (0,165,255), 'Reposo': (100,100,100)}

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w + 480, h))

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(4.8, h/100), dpi=100)
canvas = FigureCanvasAgg(fig)

i = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    
    t = i / fps
    closest = (df_pred['time_sec'] - t).abs().idxmin()
    row = df_pred.iloc[closest]
    act = row['activity']
    
    cv2.rectangle(frame, (20, 20), (650, 100), (0,0,0), -1)
    cv2.putText(frame, f'JEBI IA: {str(act).upper()}', (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_map.get(act, (255,255,255)), 3)
    
    ax.clear()
    win = df_pred[df_pred['time_sec'] <= t].tail(50)
    if not win.empty:
        ax.plot(win['time_sec'], win['acc_v_max'], color='cyan')
        ax.fill_between(win['time_sec'], 0, win['acc_v_max'], color='cyan', alpha=0.2)
    ax.set_title('Telemetría IMU')
    ax.set_ylim(0, 12)
    
    fig.tight_layout()
    canvas.draw()
    plot_img = cv2.cvtColor(np.array(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    plot_img = cv2.resize(plot_img, (480, h))
    
    combined = np.hstack((frame, plot_img))
    out.write(combined)
    if i % 1000 == 0: print(f'Procesando frame {i}...')
    i += 1

cap.release()
out.release()
"

echo "Step 5: Consolidando resultados..."
[ -f "docs/plots/preprocessing/linear_acceleration.png" ] && cp docs/plots/preprocessing/linear_acceleration.png output_run/
[ -f "docs/plots/cleaning/signal_cleaning_validation.png" ] && cp docs/plots/cleaning/signal_cleaning_validation.png output_run/
[ -f "output/cycle_details_v3.csv" ] && mv output/cycle_details_v3.csv output_run/
[ -f "output/final_predictions_v3.csv" ] && mv output/final_predictions_v3.csv output_run/

echo "------------------------------------------------"
echo "✅ Inferencia completada con éxito."
echo "Resultados en: output_run/"
echo "------------------------------------------------"
