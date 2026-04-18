# Detección Automática de Eventos

Utilizando algoritmos de umbralización adaptativa sobre la aceleración lineal y la velocidad angular, hemos identificado patrones de actividad significativos.

## Metodología
- **Impactos:** Detectados cuando la magnitud de la aceleración lineal supera los **3.5 m/s²**.
- **Giros Bruscos:** Detectados cuando la velocidad angular total supera los **1.5 rad/s**.
- **Filtro Temporal:** Se analizó la duración de cada evento para distinguir entre ruidos momentáneos y maniobras sostenidas.

## Resumen de Hallazgos
- **Total de Impactos:** 930
- **Total de Giros Bruscos:** 135

### Timeline de Actividad
![Timeline de Eventos](./plots/events/event_timeline.png)

### Análisis de Casos Notables
1.  **Maniobras de Larga Duración:** Se detectaron giros de más de 20 segundos (ej. en el segundo 26, 80, 203), lo que indica cambios de dirección prolongados.
2.  **Alta Densidad de Impactos:** Existen zonas con ráfagas de impactos cortos, posiblemente indicando una superficie irregular o una actividad repetitiva (como pasos).

## Utilidad para la IA
Estos eventos detectados por heurística sirven como **etiquetas automáticas (Weak Labels)**. Podríamos entrenar un modelo para que aprenda a clasificar estos segmentos automáticamente en el futuro.
