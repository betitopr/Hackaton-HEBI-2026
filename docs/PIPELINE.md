# Pipeline de Análisis de Datos e IA - JEBI v3.1

Este documento describe el flujo de trabajo completo del sistema JEBI, desde la ingesta de datos brutos del sensor IMU hasta la generación de KPIs de productividad.

## 1. Arquitectura General
El pipeline está dividido en tres fases críticas: Refinamiento, Entrenamiento e Inferencia Productiva.

---

## 2. Fase de Refinamiento (`scripts/pre_ai_post/01_clean_and_refine.py`)
El objetivo es transformar señales ruidosas en datos útiles para la IA.
- **Filtrado Butterworth:** Se aplica un filtro pasa-bajas de 4to orden con frecuencia de corte de 1.5Hz para eliminar las vibraciones del motor.
- **Normalización Espacial:** 
    - `acc_horiz`: $\sqrt{x^2 + y^2}$ (Giro y traslación).
    - `acc_vert`: $|z|$ (Excavación e impactos).
- **Limpieza de Etiquetas:** Normalización de etiquetas humanas y aplicación de márgenes de seguridad para evitar solapamientos en las transiciones de actividad.

## 3. Ingeniería de Características (`scripts/pre_ai_post/02_feature_generation.py`)
Transformación de series temporales en vectores de características.
- **Ventaneo (Windowing):** Ventanas de 2.0s con 75% de solapamiento.
- **Vectores de Características (10 dimensiones):**
    - Estadísticas de aceleración (Media, Desviación, Máximos).
    - Magnitud de giroscopio filtrada.
    - **Pitch:** Inclinación del brazo calculada mediante acelerometría.
    - **V/H Ratio:** Relación entre movimiento vertical y horizontal.

## 4. Entrenamiento y Regularización (`scripts/pre_ai_post/03_model_training.py`)
- **Modelo:** Random Forest Classifier.
- **Data Augmentation:** Uso de *Jittering* (ruido gaussiano del 5%) para mejorar la robustez ante diferentes tipos de terreno o máquinas.
- **Regularización:** Limitación de profundidad (`max_depth=7`) para asegurar que el modelo aprenda patrones de movimiento generales y no ruidos específicos del dataset de entrenamiento.

## 5. Inferencia y Motor de KPIs (`run_pipeline.py` & `core/analytics.py`)
Es el núcleo de ejecución del proyecto.
1. **Predicción:** Clasifica cada ventana en una de las cuatro categorías: `Carga`, `Movimiento`, `Descarga` o `Reposo`.
2. **Máquina de Estados de Ciclos:** Identifica la secuencia lógica de trabajo:
   `Carga` $\rightarrow$ `Movimiento` $\rightarrow$ `Descarga`.
3. **Cálculo de Métricas:**
    - **Tiempo de Ciclo:** Duración total desde el inicio de carga hasta el fin de descarga.
    - **Intervalos entre Ciclos:** Tiempo de retorno o espera (no productivo).
    - **Consistencia:** Desviación estándar de los tiempos de ciclo, indicando la estabilidad del operador.

---

## 6. Salidas de Datos (Artifacts)
- `data/refined/excavator_model.pkl`: Modelo entrenado.
- `output/final_predictions_v3.csv`: Clasificación temporal completa.
- `output/cycle_details_v3.csv`: Registro detallado de cada ciclo detectado para análisis de eficiencia.

---
*JEBI Documentation - 2026*
