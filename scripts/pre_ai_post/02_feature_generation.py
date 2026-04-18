import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN ---
CLEAN_IMU = 'data/refined/imu_clean.csv'
CLEAN_LABELS = 'data/refined/labels_clean.csv'
OUTPUT_FEATURES = 'data/refined/feature_set.csv'
PLOT_DIR = 'docs/plots/features'
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. CARGA DE DATOS REFINADOS
print("Cargando datos limpios...")
df = pd.read_csv(CLEAN_IMU)
labels_df = pd.read_csv(CLEAN_LABELS)

# 2. EXTRACCIÓN DE CARACTERÍSTICAS (Windowing)
# Ventana de 2 segundos (20 muestras a 10Hz)
window_size = 20 
step_size = 5 # 75% de solapamiento para tener más datos de entrenamiento

features = []

print(f"Generando características con ventanas de {window_size} muestras...")

for i in range(0, len(df) - window_size, step_size):
    win = df.iloc[i : i + window_size]
    t_mid = win['time_sec'].mean()
    
    # Calcular estadísticas sobre las señales FILTRADAS (Clean)
    # Estas son las que la IA usará para decidir
    feat = {
        'time_sec': t_mid,
        
        # Aceleración Horizontal (Giro/Traslado)
        'acc_h_mean': win['acc_horiz_clean'].mean(),
        'acc_h_std': win['acc_horiz_clean'].std(),
        'acc_h_max': win['acc_horiz_clean'].max(),
        
        # Aceleración Vertical (Excavación/Impacto)
        'acc_v_mean': win['acc_vert_clean'].mean(),
        'acc_v_std': win['acc_vert_clean'].std(),
        'acc_v_max': win['acc_vert_clean'].max(),
        
        # Giroscopio (Rotación de cabina)
        'gyr_mean': win['gyr_mag_clean'].mean(),
        'gyr_std': win['gyr_mag_clean'].std(),
        
        # Actitud (Inclinación del brazo)
        'pitch': np.arctan2(-win['linear_acc_x'], np.sqrt(win['linear_acc_y']**2 + win['linear_acc_z']**2)).mean(),
        
        # Relación Vertical/Horizontal (Diferenciador clave)
        'v_h_ratio': win['acc_vert_clean'].mean() / (win['acc_horiz_clean'].mean() + 1e-5)
    }
    features.append(feat)

feat_df = pd.DataFrame(features)

# 3. ASIGNACIÓN DE ETIQUETAS (Labeling)
def get_label(t):
    for _, row in labels_df.iterrows():
        if row['start'] <= t <= row['end']:
            # Simplificar categorías
            act = row['activity'].lower()
            if 'descarga' in act: return 'Descarga'
            if 'carga' in act: return 'Carga'
            if 'movimiento' in act: return 'Movimiento'
            if 'reposo' in act: return 'Reposo'
            return 'Otro'
    return 'Sin Etiqueta'

feat_df['label'] = feat_df['time_sec'].apply(get_label)

# 4. GRÁFICO: Distribución de Características por Actividad
print("Generando gráfico de importancia de características...")
plt.figure(figsize=(15, 10))

# Solo comparar las clases principales etiquetadas
plot_data = feat_df[feat_df['label'] != 'Sin Etiqueta']

plt.subplot(2, 2, 1)
sns.boxplot(data=plot_data, x='label', y='acc_v_std')
plt.title('Inestabilidad Vertical (acc_v_std)')
plt.ylabel('m/s²')

plt.subplot(2, 2, 2)
sns.boxplot(data=plot_data, x='label', y='gyr_mean')
plt.title('Velocidad de Giro (gyr_mean)')
plt.ylabel('rad/s')

plt.subplot(2, 2, 3)
sns.boxplot(data=plot_data, x='label', y='pitch')
plt.title('Inclinación del Brazo (pitch)')
plt.ylabel('rad')

plt.subplot(2, 2, 4)
sns.boxplot(data=plot_data, x='label', y='v_h_ratio')
plt.title('Ratio Vertical/Horizontal')

plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/feature_distribution.png')

# 5. GUARDAR DATASET MAESTRO
feat_df.to_csv(OUTPUT_FEATURES, index=False)
print(f"Dataset de características guardado en {OUTPUT_FEATURES}")
print(f"Gráfico de distribución guardado en {PLOT_DIR}/feature_distribution.png")
