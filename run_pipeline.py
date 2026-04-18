import time
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.live import Live
from rich import box

from models.random_forest_v2 import RandomForestExcavatorModel
from core.analytics import CycleAnalyzer

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]JEBI - AI INFERENCE PIPELINE v3.1[/bold cyan]\n"
            "[white]Deep Analytics & Cycle KPIs Extraction[/white]",
            box=box.DOUBLE,
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        # PASO 1: Carga de Datos
        task1 = progress.add_task("[yellow]Cargando datos IMU...", total=100)
        df_imu = pd.read_csv("data/refined/imu_clean.csv")
        progress.update(task1, completed=100)

        # PASO 2: Inicializar Modelo
        task2 = progress.add_task("[magenta]Inicializando Modelo IA...", total=100)
        ai_model = RandomForestExcavatorModel()
        ai_model.load("data/refined/excavator_model.pkl")
        progress.update(task2, completed=100)

        # PASO 3: Feature Engineering (Windowing)
        task3 = progress.add_task(
            "[cyan]Procesando Ventanas Temporales...", total=len(df_imu)
        )
        window_size = 20
        step_size = 5
        features = []

        for i in range(0, len(df_imu) - window_size, step_size):
            win = df_imu.iloc[i : i + window_size]
            feat = {
                "time_sec": win["time_sec"].mean(),
                "acc_h_mean": win["acc_horiz_clean"].mean(),
                "acc_h_std": win["acc_horiz_clean"].std(),
                "acc_h_max": win["acc_horiz_clean"].max(),
                "acc_v_mean": win["acc_vert_clean"].mean(),
                "acc_v_std": win["acc_vert_clean"].std(),
                "acc_v_max": win["acc_vert_clean"].max(),
                "gyr_mean": win["gyr_mag_clean"].mean(),
                "gyr_std": win["gyr_mag_clean"].std(),
                "pitch": np.arctan2(
                    -win["linear_acc_x"],
                    np.sqrt(win["linear_acc_y"] ** 2 + win["linear_acc_z"] ** 2),
                ).mean(),
                "v_h_ratio": win["acc_vert_clean"].mean()
                / (win["acc_horiz_clean"].mean() + 1e-5),
            }
            features.append(feat)
            if i % 100 == 0:
                progress.update(task3, advance=100)

        df_features = pd.DataFrame(features)
        progress.update(task3, completed=len(df_imu))

        # PASO 4: Inferencia
        task4 = progress.add_task("[green]Realizando Predicciones...", total=100)
        predictions = ai_model.predict(df_features)
        df_features["activity"] = predictions
        progress.update(task4, completed=100)

        # PASO 5: Análisis de Ciclos y Métricas
        task5 = progress.add_task(
            "[yellow]Calculando KPIs de Productividad...", total=100
        )
        analyzer = CycleAnalyzer(df_features)
        metrics_res = analyzer.extract_metrics()
        progress.update(task5, completed=100)

    # RESULTADOS FINALES EN TABLA
    console.print("\n[bold green]✓ Procesamiento Completado[/bold green]\n")

    # --- TABLA 1: ACTIVIDADES GENERALES ---
    summary = df_features["activity"].value_counts()
    table1 = Table(title="Resumen de Actividades (Time Share)", box=box.ROUNDED)
    table1.add_column("Actividad", style="cyan")
    table1.add_column("Tiempo Est.", justify="right", style="green")
    table1.add_column("Porcentaje", justify="right", style="magenta")

    total_samples = len(df_features)
    for act, count in summary.items():
        table1.add_row(
            act, f"{count * 0.5:.1f}s", f"{(count / total_samples) * 100:.1f}%"
        )

    console.print(table1)

    # --- TABLA 2: KPIs DE CICLOS ---
    if metrics_res:
        metrics_summary, cycles_df = metrics_res
        table2 = Table(
            title="KPIs de Ciclos de Trabajo (Deep Analytics)", box=box.HEAVY_HEAD
        )
        table2.add_column("Métrica", style="white")
        table2.add_column("Valor", justify="right", style="bold yellow")
        table2.add_column("Unidad", style="dim")

        table2.add_row(
            "Total de Ciclos Completos", f"{metrics_summary['total_cycles']}", "ciclos"
        )
        table2.add_row(
            "Tiempo de Ciclo Promedio",
            f"{metrics_summary['avg_cycle_time']:.2f}",
            "seg/ciclo",
        )
        table2.add_row(
            "Intervalo entre Ciclos (Espera)",
            f"{metrics_summary['avg_interval']:.2f}",
            "seg",
        )
        table2.add_row(
            "Consistencia Operativa (StdDev)",
            f"{metrics_summary['std_cycle_time']:.2f}",
            "seg",
        )
        table2.add_row(
            "Tiempo Productivo Neto",
            f"{metrics_summary['total_productive_time']:.1f}",
            "seg",
        )

        console.print("\n", table2)

        # GUARDAR RESULTADOS DETALLADOS
        cycles_df.to_csv("output/cycle_details_v3.csv", index=False)
        console.print(
            f"\n[dim]Detalle de ciclos guardado en: output/cycle_details_v3.csv[/dim]"
        )
    else:
        console.print(
            "\n[red]No se detectaron ciclos completos suficientes para el análisis.[/red]"
        )

    # GUARDAR PREDICCIONES
    df_features.to_csv("output/final_predictions_v3.csv", index=False)
    console.print(
        f"[dim]Predicciones globales guardadas en: output/final_predictions_v3.csv[/dim]"
    )


if __name__ == "__main__":
    main()
