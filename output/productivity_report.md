# 📊 Reporte de Productividad de Excavadora (IA Powered v2.0)

**Duración del Análisis:** 14.96 minutos
**Precisión del Modelo:** 80.0% (Enriquecido con Jerk y Euler Angles)

## 🚀 Resumen de Eficiencia (Timeline Analizado)
| Actividad | % del Tiempo | Clasificación |
| :--- | :---: | :--- |
| **Movimiento** | 60.3% | ⚠️ Logístico |
| **Descarga** | 24.7% | ✅ Productivo |
| **Carga** | 14.1% | ✅ Productivo |
| **Reposo** | 0.5% | ❌ Improductivo |
| **Otro** | 0.4% | ⚠️ Logístico |

## 🚜 Métricas de Operación Crítica
- **Total de Ciclos detectados:** 29 (Aproximadamente 116 ciclos/hora).
- **Impactos de Alta Intensidad:** 930 (Detección por umbral de aceleración lineal).
- **Maniobras de Giro Brusco:** 135 (Detección por velocidad angular).
- **Sincronización:** 100% (9403 muestras IMU alineadas con 9403 frames de video).

## 💡 Insights de Ingeniería y Optimización
- **FIRMA MECÁNICA:** El análisis FFT reveló una frecuencia dominante de **0.03 Hz**, confirmando movimientos de ciclo largo y descartando vibraciones parasitarias de alta frecuencia.
- **ESTABILIDAD:** La inclinación promedio (**Pitch**) se mantuvo estable en **-43 grados**, validando una postura de operación consistente.
- **ALERTA DE IMPACTO:** El pico máximo de carga alcanzó los **47.85 m/s²**, lo que sugiere maniobras de alta potencia en el segundo 392.

---
*Reporte generado automáticamente por el Pipeline de Análisis JEBI.*
