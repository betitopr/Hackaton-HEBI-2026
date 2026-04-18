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

## Segmentación de Video (Highlights)

Se han extraído clips de video de 4 segundos centrados en los impactos más intensos para validación visual:

| Ranking | Tiempo (s) | Magnitud (m/s²) | Archivo Generado |
| :--- | :--- | :--- | :--- |
| 1 | 392.74 | 47.85 | `impact_top_1_392s.mp4` |
| 2 | 862.47 | 35.73 | `impact_top_2_862s.mp4` |
| 3 | 258.87 | 34.34 | `impact_top_3_258s.mp4` |
| 4 | 342.87 | 33.19 | `impact_top_4_342s.mp4` |
| 5 | 14.13 | 27.67 | `impact_top_5_14s.mp4` |

*Nota: Estos clips pueden generarse utilizando el script `scripts/video_segmentation.py`.*

## Utilidad para la IA
Estos eventos detectados por heurística sirven como **etiquetas automáticas (Weak Labels)**. Podríamos entrenar un modelo para que aprenda a clasificar estos segmentos automáticamente en el futuro.
