import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# --- CONFIGURACIÓN ---
FEATURE_SET = 'data/refined/imu_clean.csv'
MODEL_FILE = 'data/refined/excavator_model.pkl'
REPORT_OUTPUT = 'output/productivity_report.md'
PLOT_DIR = 'docs/plots/inference'
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. CARGA DE MODELO Y DATOS
print("Cargando modelo de IA y dataset de características...")
model = joblib.load(MODEL_FILE)
df_feat = pd.read_csv(FEATURE_SET)

# 2. PREPARACIÓN DE CARACTERÍSTICAS (Windowing igual al entrenamiento)
window_size = 20 
step_size = 5    

features = []
print("Generando ventanas para inferencia global...")
for i in range(0, len(df_feat) - window_size, step_size):
    win = df_feat.iloc[i : i + window_size]
    t_mid = win['time_sec'].mean()
    
    # IMPORTANTE: Estas deben ser IDÉNTICAS a 02_feature_generation.py
    feat = {
        'time_sec': t_mid,
        'acc_h_mean': win['acc_horiz_clean'].mean(),
        'acc_h_std': win['acc_horiz_clean'].std(),
        'acc_h_max': win['acc_horiz_clean'].max(),
        'acc_v_mean': win['acc_vert_clean'].mean(),
        'acc_v_std': win['acc_vert_clean'].std(),
        'acc_v_max': win['acc_vert_clean'].max(),
        'gyr_mean': win['gyr_mag_clean'].mean(),
        'gyr_std': win['gyr_mag_clean'].std(),
        'pitch': np.arctan2(-win['linear_acc_x'], np.sqrt(win['linear_acc_y']**2 + win['linear_acc_z']**2)).mean(),
        'v_h_ratio': win['acc_vert_clean'].mean() / (win['acc_horiz_clean'].mean() + 1e-5)
    }
    features.append(feat)

df_inf = pd.DataFrame(features)
X = df_inf.drop(['time_sec'], axis=1)

# 3. INFERENCIA
print("Realizando predicción global...")
df_inf['raw_pred'] = model.predict(X)

# 4. POST-PROCESAMIENTO: Suavizado
def smooth_labels(series, window=5):
    smoothed = series.copy()
    for i in range(len(series)):
        start = max(0, i - window // 2)
        end = min(len(series), i + window // 2 + 1)
        win = series[start:end]
        smoothed.iloc[i] = win.value_counts().index[0]
    return smoothed

df_inf['smooth_pred'] = smooth_labels(df_inf['raw_pred'], window=5)
df_inf.to_csv('data/activity_predictions.csv', index=False) # Actualizar para el video

# 5. CÁLCULO DE MÉTRICAS
total_time = df_inf['time_sec'].iloc[-1] - df_inf['time_sec'].iloc[0]
counts = df_inf['smooth_pred'].value_counts(normalize=True) * 100

# Identificación de Ciclos (Máquina de Estados)
# Ciclo Ideal: CARGA -> MOVIMIENTO -> DESCARGA
print("Analizando ciclos de trabajo (Máquina de Estados)...")
state_sequence = []
last_s = None
for s in df_inf['smooth_pred']:
    if s != last_s:
        state_sequence.append(s)
        last_s = s

cycles = 0
in_cycle = False
has_moved = False

for i in range(len(state_sequence)):
    s = state_sequence[i]
    if s == 'Carga':
        in_cycle = True
        has_moved = False
    elif s == 'Movimiento' and in_cycle:
        has_moved = True
    elif s == 'Descarga' and in_cycle and has_moved:
        cycles += 1
        in_cycle = False
        has_moved = False
    elif s == 'Reposo':
        in_cycle = False 

# 5. GENERACIÓN DEL REPORTE (Markdown)
# ... (igual que antes)
# 6. VISUALIZACIÓN FINAL
plt.figure(figsize=(20, 4))
color_map = {'Carga': 'red', 'Descarga': 'green', 'Movimiento': 'orange', 'Reposo': 'gray', 'Otro': 'blue'}

for act in df_inf['smooth_pred'].unique():
    subset = df_inf[df_inf['smooth_pred'] == act]
    plt.scatter(subset['time_sec'], [1]*len(subset), color=color_map.get(act, 'black'), label=act, s=100, marker='|')

plt.title('Timeline de Operación Real de la Excavadora (Predicho por IA)')
plt.xlabel('Tiempo (s)')
plt.yticks([])
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/productivity_timeline.png')

print(f"Reporte generado en {REPORT_OUTPUT}")
print(f"Timeline final guardado en {PLOT_DIR}/productivity_timeline.png")
