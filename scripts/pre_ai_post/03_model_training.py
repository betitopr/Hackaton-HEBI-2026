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

# 2. PREPARACIÓN DE VARIABLES Y DATA AUGMENTATION
X = train_data.drop(['time_sec', 'label'], axis=1)
y = train_data['label']

# --- ESTRATEGIA ANTI-OVERFITTING: Jittering (Ruido Gaussiano) ---
print("Aplicando Data Augmentation (Jittering) para reducir overfitting...")
X_noisy = X + np.random.normal(0, X.std() * 0.05, X.shape)
X_combined = pd.concat([X, X_noisy], axis=0)
y_combined = pd.concat([y, y], axis=0)

# División en Entrenamiento y Prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# 3. ENTRENAMIENTO CON REGULARIZACIÓN
print("Entrenando clasificador con regularización agresiva...")
model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=7,            # Reducido de 10 a 7 para generalizar mejor
    min_samples_leaf=5,     # Evita hojas con muy pocos datos (ruido)
    max_features='sqrt',    # Fuerza a los árboles a usar subconjuntos de features
    random_state=42,
    n_jobs=-1               # Usar todos los núcleos
)
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
