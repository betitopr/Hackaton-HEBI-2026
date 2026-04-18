import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# --- LOGICA HIERARCHICAL (No Hardcoded) ---
class ExcavatorHybridModel:
    def __init__(self, rf_model, thresholds):
        self.rf = rf_model
        self.thresholds = thresholds
        
    def predict(self, X_row):
        # 1. Regla Física: Si hay mucho giro, es movimiento
        if X_row['gyr_mean'] > self.thresholds['gyro_move']:
            return 'Movimiento'
        
        # 2. Regla Física: Si no hay casi movimiento vertical ni horizontal, es reposo
        if X_row['acc_v_std'] < self.thresholds['acc_reposo'] and X_row['acc_h_std'] < self.thresholds['acc_reposo']:
            return 'Reposo'
            
        # 3. Inteligencia Artificial: Para lo difícil (Carga vs Descarga)
        # Convertir fila a formato que espera sklearn
        return self.rf.predict([X_row.drop('time_sec', errors='ignore')])[0]

# --- ENTRENAMIENTO ROBUSTO ---
def train_hierarchical():
    print("Cargando dataset y etiquetas robustas...")
    df = pd.read_csv('data/refined/feature_set.csv')
    robust_labels = pd.read_csv('data/refined/labels_robust.csv')
    
    # Solo entrenar con datos que están DENTRO de los rangos robustos (sin ruido de 1s)
    def is_robust(t):
        for _, row in robust_labels.iterrows():
            if row['start'] <= t <= row['end']:
                return row['label']
        return None

    df['robust_label'] = df['time_sec'].apply(is_robust)
    train_df = df[df['robust_label'].notnull()].copy()
    
    # Limpiar nombres de etiquetas (normalizar)
    def normalize(l):
        l = l.lower()
        if 'carga' in l: return 'Carga'
        if 'descarga' in l: return 'Descarga'
        if 'movimiento' in l: return 'Movimiento'
        if 'reposo' in l: return 'Reposo'
        return 'Otro'
    
    train_df['target'] = train_df['robust_label'].apply(normalize)
    
    X = train_df.drop(['time_sec', 'label', 'robust_label', 'target'], axis=1)
    y = train_df['target']
    
    print(f"Entrenando con {len(X)} muestras PURAS (sin ruido de transición)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
    rf.fit(X, y)
    
    # Calcular umbrales dinámicos basados en los datos de entrenamiento
    # Para que no estén hardcodeados
    thresholds = {
        'gyro_move': train_df[train_df['target'] == 'Movimiento']['gyr_mean'].quantile(0.2), # El 20% más bajo de movimiento
        'acc_reposo': train_df[train_df['target'] == 'Reposo']['acc_v_std'].quantile(0.8)   # El 80% más alto de reposo
    }
    
    print(f"Umbrales dinámicos calculados: {thresholds}")
    
    return rf, thresholds

if __name__ == "__main__":
    rf_model, thresholds = train_hierarchical()
    # Guardar
    joblib.dump({'model': rf_model, 'thresholds': thresholds}, 'data/refined/hybrid_model.pkl')
    print("Modelo Híbrido Guardado.")
