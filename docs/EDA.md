# Análisis Exploratorio de Datos (EDA) - IMU

Este documento detalla la estructura y el contenido de los datos inerciales (IMU) encontrados en el archivo `.npy`.

## Estructura del Archivo
El archivo `data/40343737_20260313_110600_to_112100_imu.npy` contiene una matriz de **9403 filas** y **11 columnas**.

### Definición de Columnas
A partir del análisis de los datos, se ha determinado la siguiente estructura:

| Índice | Nombre | Unidad | Descripción |
| :--- | :--- | :--- | :--- |
| 0 | `timestamp` | ns | Tiempo en nanosegundos (Unix Epoch). |
| 1 | `accel_x` | m/s² | Aceleración en el eje X. |
| 2 | `accel_y` | m/s² | Aceleración en el eje Y. |
| 3 | `accel_z` | m/s² | Aceleración en el eje Z. |
| 4 | `gyro_x` | rad/s | Velocidad angular en el eje X. |
| 5 | `gyro_y` | rad/s | Velocidad angular en el eje Y. |
| 6 | `gyro_z` | rad/s | Velocidad angular en el eje Z. |
| 7 | `quat_w` | - | Componente W (escalar) del cuaternión de orientación. |
| 8 | `quat_x` | - | Componente X del cuaternión de orientación. |
| 9 | `quat_y` | - | Componente Y del cuaternión de orientación. |
| 10 | `quat_z` | - | Componente Z del cuaternión de orientación. |

## Visualización
A continuación se muestra la serie temporal de los sensores:

![Análisis IMU](./plots/imu_analysis.png)

### Observaciones Clave:
- **Gravedad:** El eje `accel_y` muestra un valor base cercano a -9.8 m/s², lo que indica que el eje Y está alineado verticalmente con la gravedad en la posición de reposo.
- **Ruidos y Eventos:** Se observan picos de aceleración que sugieren movimientos bruscos o vibraciones durante la captura.
- **Orientación:** Los cuaterniones son estables, permitiendo una reconstrucción precisa de la actitud del dispositivo.

## Sincronización de Video
Existen archivos de video (`left.mp4` y `right.mp4`) que corresponden al mismo periodo de tiempo. El siguiente paso será alinear los timestamps del IMU con los frames de los videos para un análisis multi-modal.
