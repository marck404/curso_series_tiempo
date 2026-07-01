#!/usr/bin/env python3
"""Visualizacion de forecasting con seaborn para train, test y predicciones.

Ejemplo:
python visualize_forecast.py \
  --train-file datasets/DailyDelhiClimateTrain.csv \
  --test-file datasets/DailyDelhiClimateTest.csv \
  --pred-file results/predictions_xgb.csv \
  --output-dir results/plots \
  --model-name xgb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera graficas bonitas de predicciones de series de tiempo con seaborn."
    )
    parser.add_argument("--train-file", required=True, type=str, help="Ruta al CSV de train")
    parser.add_argument("--test-file", required=True, type=str, help="Ruta al CSV de test")
    parser.add_argument("--pred-file", required=True, type=str, help="Ruta al CSV de predicciones")
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        type=str,
        help="Directorio de salida para la figura",
    )
    parser.add_argument(
        "--target",
        default="meantemp",
        type=str,
        help="Variable objetivo en train y test",
    )
    parser.add_argument(
        "--model-name",
        default="modelo",
        type=str,
        help="Nombre del modelo para titulos",
    )
    parser.add_argument(
        "--dpi",
        default=220,
        type=int,
        help="Resolucion de salida de la imagen",
    )
    return parser.parse_args()


def load_data(train_path: Path, test_path: Path, pred_path: Path, target: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    pred_df = pd.read_csv(pred_path)

    for df in (train_df, test_df, pred_df):
        if "date" not in df.columns:
            raise ValueError("Todos los archivos deben incluir columna date")

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"La columna objetivo {target} no existe en train/test")

    required_pred_cols = {"date", "y_true", "y_pred", "error"}
    missing_pred = required_pred_cols - set(pred_df.columns)
    if missing_pred:
        raise ValueError(f"Faltan columnas en predicciones: {sorted(missing_pred)}")

    train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce")
    test_df["date"] = pd.to_datetime(test_df["date"], errors="coerce")
    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")

    train_df = train_df.sort_values("date").reset_index(drop=True)
    test_df = test_df.sort_values("date").reset_index(drop=True)
    pred_df = pred_df.sort_values("date").reset_index(drop=True)

    merged = pred_df.merge(
        test_df[["date", target]].rename(columns={target: "target_test"}),
        on="date",
        how="left",
    )

    return train_df, test_df, merged


def build_plot(train_df, test_df, merged_df, target: str, model_name: str):
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("deep")

    colors = {
        "train": "#2A9D8F",
        "test": "#E9C46A",
        "pred": "#E76F51",
        "resid": "#457B9D",
    }

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    fig.patch.set_facecolor("#F7F7F7")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#FCFCFC")

    # Panel principal: toda la serie con zona de test resaltada.
    sns.lineplot(
        data=train_df,
        x="date",
        y=target,
        ax=ax1,
        color=colors["train"],
        linewidth=2.2,
        label="Train",
    )
    sns.lineplot(
        data=test_df,
        x="date",
        y=target,
        ax=ax1,
        color=colors["test"],
        linewidth=2.2,
        label="Test real",
    )
    sns.lineplot(
        data=merged_df,
        x="date",
        y="y_pred",
        ax=ax1,
        color=colors["pred"],
        linewidth=2.5,
        linestyle="--",
        label=f"Prediccion {model_name}",
    )

    test_start = test_df["date"].min()
    test_end = test_df["date"].max()
    ax1.axvspan(test_start, test_end, color="#FFE8B6", alpha=0.25, label="Ventana test")

    ax1.set_title(
        f"Pronostico de {target} - Train, Test y Prediccion ({model_name})",
        fontsize=18,
        pad=12,
        weight="bold",
    )
    ax1.set_xlabel("Fecha")
    ax1.set_ylabel(target)
    ax1.legend(loc="upper left", frameon=True, fancybox=True)

    # Panel inferior izquierdo: zoom en test con banda de error absoluto.
    sns.lineplot(
        data=merged_df,
        x="date",
        y="y_true",
        ax=ax2,
        color=colors["test"],
        linewidth=2.0,
        label="Real",
    )
    sns.lineplot(
        data=merged_df,
        x="date",
        y="y_pred",
        ax=ax2,
        color=colors["pred"],
        linewidth=2.2,
        linestyle="--",
        label="Prediccion",
    )
    ax2.fill_between(
        merged_df["date"],
        merged_df["y_true"],
        merged_df["y_pred"],
        color="#F4A261",
        alpha=0.18,
        label="Error",
    )
    ax2.set_title("Zoom test: real vs prediccion", fontsize=14, weight="bold")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel(target)
    ax2.legend(loc="upper left", frameon=True)

    # Panel inferior derecho: residuos en tiempo + distribucion.
    sns.scatterplot(
        data=merged_df,
        x="date",
        y="error",
        ax=ax3,
        color=colors["resid"],
        s=45,
        alpha=0.85,
        label="Residuo",
    )
    sns.lineplot(
        data=merged_df,
        x="date",
        y="error",
        ax=ax3,
        color="#1D3557",
        linewidth=1.4,
        alpha=0.5,
        legend=False,
    )
    ax3.axhline(0, color="#D62828", linestyle=":", linewidth=2)
    ax3.set_title("Residuos en el tiempo (y_true - y_pred)", fontsize=14, weight="bold")
    ax3.set_xlabel("Fecha")
    ax3.set_ylabel("Error")
    ax3.legend(loc="upper left", frameon=True)

    sns.despine(fig=fig)
    return fig


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    pred_path = Path(args.pred_file)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, merged_df = load_data(train_path, test_path, pred_path, args.target)

    fig = build_plot(train_df, test_df, merged_df, args.target, args.model_name)
    out_path = output_dir / f"forecast_visual_{args.model_name}.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Visualizacion generada correctamente")
    print(f"Archivo: {out_path}")


if __name__ == "__main__":
    main()
