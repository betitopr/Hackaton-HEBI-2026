import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import os

# 1. Parser de Etiquetas Ultra-Robusto
def parse_labels(file_path):
    labels = []
    current_min = 0
    last_sec = -1
    
    # Mapeo de corrección de errores comunes y normalización
    def normalize_act(text):
        text = text.lower()
        if 'decsarga' in text or 'descarga' in text: return 'Descarga'
        if 'carga' in text: return 'Carga'
        if 'movi' in text or 'mueve' in text: return 'Movimiento'
        if 'reposo' in text or 'esperando' in text: return 'Reposo'
        return 'Otro'

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line: continue
            
            # Formato Rango: 00:00 - 00:04 -> Actividad
            match_range = re.match(r'(\d+):(\d+)\s*-\s*(\d+):(\d+)\s*->\s*(.*)', line)
            if match_range:
                s_min, s_sec, e_min, e_sec, act = match_range.groups()
                start = int(s_min) * 60 + int(s_sec)
                end = int(e_min) * 60 + int(e_sec)
                labels.append((start, end, normalize_act(act)))
                current_min = int(e_min)
                last_sec = int(e_sec)
                continue
            
            # Formato Simple: 27 -> Actividad
            match_sec = re.match(r'(\d+)\s*->\s*(.*)', line)
            if match_sec:
                sec = int(match_sec.group(1))
                # Detectar cambio de minuto automático
                if sec < last_sec: 
                    current_min += 1
                
                time_abs = current_min * 60 + sec
                act = normalize_act(match_sec.group(2))
                
                # Cerrar el hueco de la etiqueta anterior si existe
                if labels:
                    ls, le, la = labels[-1]
                    labels[-1] = (ls, time_abs, la)
                
                # Asumir que esta nueva actividad dura al menos 2 seg hasta el próximo reporte
                labels.append((time_abs, time_abs + 2, act))
                last_sec = sec
                
    return labels

# 2. Carga de Datos Enriquecidos
df = pd.read_csv('data/imu_enriched.csv')

# 3. Ventaneo Fino (1 segundo de ventana para capturar Descargas)
window_size_samples = 10 # 1 segundo a 10Hz
step_size_samples = 2    # Alta resolución (80% overlap)

features = []
print("Generando características con resolución fina...")

for i in range(0, len(df) - window_size_samples, step_size_samples):
    win = df.iloc[i : i + window_size_samples]
    t_mid = win['time_sec'].mean()
    
    feat = {
        'time_sec': t_mid,
        'acc_std': win['linear_mag'].std(),
        'acc_mean': win['linear_mag'].mean(),
        'acc_max': win['linear_mag'].max(),
        'gyr_std': win['gyr_mag'].std(),
        'gyr_max': win['gyr_mag'].max(),
        'jerk_std': win['jerk'].std(),
        'jerk_max': win['jerk'].max(),
        'roll_std': win['roll'].std(),
        'pitch_std': win['pitch'].std(),
        'yaw_std': win['yaw'].std(),
        'pitch_mean': win['pitch'].mean()
    }
    features.append(feat)

feat_df = pd.DataFrame(features)

# 4. Mapeo de Etiquetas
human_labels = parse_labels('labels.txt')
print(f"Etiquetas humanas procesadas: {len(human_labels)}")

def get_label(t):
    for s, e, act in human_labels:
        if s <= t <= e: return act
    return None

feat_df['label'] = feat_df['time_sec'].apply(get_label)

# 5. Entrenamiento Balanceado
train_data = feat_df.dropna(subset=['label'])
# Limpiar clases con poquísimos datos si es necesario, pero intentaremos con todas primero
X = train_data.drop(['time_sec', 'label'], axis=1)
y = train_data['label']

print(f"Distribución de clases en entrenamiento:\n{y.value_counts()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Usamos Balanced Random Forest para compensar el desbalance
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# 6. Evaluación y Guardado
y_pred = clf.predict(X_test)
print("\n--- REPORTE DE CLASIFICACIÓN FINAL ---")
print(classification_report(y_test, y_pred))

# Guardar
joblib.dump(clf, 'data/activity_model.pkl')
feat_df['pred_label'] = clf.predict(feat_df.drop(['time_sec', 'label'], axis=1))
feat_df.to_csv('data/activity_predictions.csv', index=False)

# Visualizar Matriz de Confusión para ver dónde falla
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Matriz de Confusión - Errores de la IA')
plt.savefig('docs/plots/confusion_matrix.png')
