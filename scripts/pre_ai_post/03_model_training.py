import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- CONFIGURACIÓN ---
FEATURE_SET = 'data/refined/feature_set.csv'
MODEL_OUTPUT = 'data/refined/excavator_model.pkl'
PLOT_DIR = 'docs/plots/training'
os.makedirs(PLOT_DIR, exist_ok=True)

# 1. CARGA DE DATOS
print("Cargando dataset de características...")
df = pd.read_csv(FEATURE_SET)

# Filtrar solo las ventanas que tienen etiqueta humana real
train_data = df[df['label'] != 'Sin Etiqueta'].copy()
print(f"Total de ventanas para entrenamiento: {len(train_data)}")

# 2. PREPARACIÓN DE VARIABLES
# Seleccionamos las características, excluyendo el tiempo y la etiqueta
X = train_data.drop(['time_sec', 'label'], axis=1)
y = train_data['label']

# División en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. ENTRENAMIENTO
print("Entrenando clasificador Random Forest...")
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 4. EVALUACIÓN
print("\n--- REPORTE DE RENDIMIENTO ---")
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# 5. VISUALIZACIÓN: Matriz de Confusión
print("Generando matriz de confusión...")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Matriz de Confusión: Predicción vs Realidad')
plt.ylabel('Realidad (Etiqueta Humana)')
plt.xlabel('Predicción (IA)')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/confusion_matrix.png')

# 6. VISUALIZACIÓN: Importancia de Características
print("Generando gráfico de importancia de variables...")
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importances.plot(kind='bar', color='teal')
plt.title('¿En qué se fija más la IA para decidir?')
plt.ylabel('Importancia Relativa')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/feature_importance.png')

# 7. GUARDAR MODELO
joblib.dump(model, MODEL_OUTPUT)
print(f"\nModelo guardado en {MODEL_OUTPUT}")
print(f"Métricas visuales guardadas en {PLOT_DIR}/")
