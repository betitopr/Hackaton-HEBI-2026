# Pre-procesamiento e Ingeniería de Características

En esta fase, transformamos los datos crudos del sensor en señales físicas interpretables, eliminando artefactos ambientales como la gravedad.

## Extracción de Aceleración Lineal

El acelerómetro mide tanto el movimiento dinámico como la aceleración gravitacional. Para aislar el movimiento real, seguimos estos pasos:

1.  **Transformación de Coordenadas:** Usando los cuaterniones de orientación, rotamos los vectores de aceleración desde el "Marco del Dispositivo" al "Marco del Mundo".
2.  **Calibración de Gravedad:** Identificamos el vector de gravedad constante en el mundo (aproximadamente -9.55 m/s² en el eje Y global) y lo sustraemos de toda la serie temporal.
3.  **Resultado:** Obtenemos la **Aceleración Lineal**, que es cero cuando el dispositivo está en reposo, sin importar su inclinación.

### Visualización
![Aceleración Lineal](./plots/preprocessing/linear_acceleration.png)

## Beneficios del Procesamiento
- **Magnitud de Movimiento Puro:** La gráfica de magnitud ahora muestra picos claros que corresponden a impactos o aceleraciones reales, eliminando el "ruido" de la gravedad que antes contaminaba la señal.
- **Independencia de Orientación:** Ahora podemos comparar movimientos ocurridos en diferentes momentos de la captura, incluso si el dispositivo cambió su inclinación.

## Archivos Generados
- `data/imu_processed.csv`: Contiene todas las columnas originales más las nuevas columnas de aceleración lineal y magnitud.
- `scripts/linear_accel_extraction.py`: Script reproducible para generar estos datos.
