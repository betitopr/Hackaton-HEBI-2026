import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

def evaluate_stability():
    print("Cargando modelos para comparativa...")
    old_model = joblib.load('data/refined/excavator_model.pkl')
    hybrid_data = joblib.load('data/refined/hybrid_model.pkl')
    rf_hybrid = hybrid_data['model']
    thresholds = hybrid_data['thresholds']
    
    df = pd.read_csv('data/refined/feature_set.csv')
    
    # Seleccionar un tramo del video que NO se usó para entrenar (ej. del segundo 200 al 400)
    test_df = df[(df['time_sec'] > 200) & (df['time_sec'] < 400)].copy()
    
    # 1. Predicciones Modelo Viejo
    X_old = test_df.drop(['time_sec', 'label'], axis=1)
    test_df['pred_old'] = old_model.predict(X_old)
    
    # 2. Predicciones Modelo Híbrido Robusto
    def predict_hybrid(row):
        if row['gyr_mean'] > thresholds['gyro_move']: return 'Movimiento'
        if row['acc_v_std'] < thresholds['acc_reposo'] and row['acc_h_std'] < thresholds['acc_reposo']: return 'Reposo'
        
        # IA solo para casos ambiguos
        X_row = row.drop(['time_sec', 'label', 'pred_old'], errors='ignore')
        return rf_hybrid.predict([X_row])[0]
        
    test_df['pred_hybrid'] = test_df.apply(predict_hybrid, axis=1)
    
    # CALCULO DE RESULTADOS: MÉTRICA DE ESTABILIDAD
    # Contamos cuántas veces cambia de estado por minuto
    def count_switches(series):
        return (series != series.shift()).sum()
    
    switches_old = count_switches(test_df['pred_old'])
    switches_hybrid = count_switches(test_df['pred_hybrid'])
    
    print("\n" + "="*40)
    print("RESULTADOS DE LA PRUEBA DE GENERALIZACIÓN")
    print("="*40)
    print(f"Cambios erráticos (Modelo Viejo): {switches_old}")
    print(f"Cambios erráticos (Modelo Híbrido): {switches_hybrid}")
    print("-" * 40)
    
    improvement = (switches_old - switches_hybrid) / switches_old * 100
    print(f"MEJORA EN ESTABILIDAD: {improvement:.1f}%")
    
    if switches_hybrid < switches_old:
        print("CONCLUSIÓN: El modelo híbrido es más robusto y tiene menos overfitting.")
    else:
        print("CONCLUSIÓN: Se requiere más limpieza de datos.")
    print("="*40)

if __name__ == "__main__":
    evaluate_stability()
