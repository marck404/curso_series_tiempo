#!/usr/bin/env python3
"""CLI de forecasting para `meantemp` con Suavizado Exponencial y XGBoost.

Uso rapido:
python forecast_climate.py \
  --train-file datasets/DailyDelhiClimateTrain.csv \
  --test-file datasets/DailyDelhiClimateTest.csv \
  --models exsm xgb \
  --output-dir results
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ForecastArtifacts:
    model_name: str
    predictions: pd.DataFrame
    metrics: Dict[str, float]
    model_path: Path


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("lag_"):
        return "lag"
    if feature_name.startswith("roll_mean_"):
        return "rolling_mean"
    return "calendar"


def extract_feature_importances(model, feature_names: Sequence[str]) -> Dict[str, float]:
    if not hasattr(model, "feature_importances_"):
        return {}

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {}

    return {
        feature_name: float(importance)
        for feature_name, importance in zip(feature_names, importances)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecasting de series de tiempo para DailyDelhiClimate."
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Ruta al CSV de entrenamiento.",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Ruta al CSV de prueba.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="meantemp",
        help="Variable objetivo a pronosticar (default: meantemp).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["exsm", "xgb"],
        default=["exsm", "xgb"],
        help="Modelos a ejecutar (exsm, xgb).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directorio de salida para metricas, predicciones y modelos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 7, 14, 30, 364, 365, 366, 367],
        help="Lista de lags para crear features autoregresivas. Incluye lags anuales (364-367) para capturar seasonality.",
    )
    parser.add_argument(
        "--rolling-windows",
        nargs="+",
        type=int,
        default=[7, 30, 365],
        help="Ventanas para medias moviles de la variable objetivo. Incluye ventana anual (365) para seasonality.",
    )
    parser.add_argument(
        "--exsm-seasonal-periods",
        type=int,
        default=365,
        help="Periodo estacional para Exponential Smoothing (default: 365 dias).",
    )
    parser.add_argument(
        "--exsm-trend",
        type=str,
        choices=["add", "mul", None],
        default="add",
        help="Tipo de tendencia para Exponential Smoothing (add=aditiva, mul=multiplicativa, None=sin tendencia).",
    )
    parser.add_argument(
        "--exsm-seasonal",
        type=str,
        choices=["add", "mul", None],
        default="add",
        help="Tipo de estacionalidad para Exponential Smoothing (add=aditiva, mul=multiplicativa, None=sin estacionalidad).",
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=240,
        help="Numero de estimadores para XGBoost.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=4,
        help="Profundidad maxima de arbol para XGBoost.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate para XGBoost.",
    )

    return parser.parse_args()


def load_dataset(path: Path, target: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    df = pd.read_csv(path)
    required = {"date", target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {path}: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError(f"Hay fechas invalidas en {path}")

    # Imputacion simple para evitar que nulos rompan la ingenieria de lags.
    if df[target].isna().any():
        df[target] = df[target].interpolate(limit_direction="both")

    return df


def build_supervised_matrix(
    series: pd.Series,
    dates: pd.Series,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> Tuple[pd.DataFrame, pd.Series]:
    work = pd.DataFrame({"date": dates, "y": series}).copy()

    for lag in lags:
        work[f"lag_{lag}"] = work["y"].shift(lag)

    for window in rolling_windows:
        work[f"roll_mean_{window}"] = work["y"].shift(1).rolling(window=window).mean()

    work["dayofweek"] = work["date"].dt.dayofweek
    work["month"] = work["date"].dt.month
    work["dayofyear"] = work["date"].dt.dayofyear
    work["weekofyear"] = work["date"].dt.isocalendar().week.astype(int)

    work = work.dropna().reset_index(drop=True)
    feature_cols = [c for c in work.columns if c not in {"date", "y"}]
    X = work[feature_cols]
    y = work["y"]

    return X, y


def make_feature_row(
    history: Sequence[float],
    date_value: pd.Timestamp,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> Dict[str, float]:
    row: Dict[str, float] = {}

    for lag in lags:
        row[f"lag_{lag}"] = float(history[-lag])

    for window in rolling_windows:
        row[f"roll_mean_{window}"] = float(np.mean(history[-window:]))

    row["dayofweek"] = int(date_value.dayofweek)
    row["month"] = int(date_value.month)
    row["dayofyear"] = int(date_value.dayofyear)
    row["weekofyear"] = int(date_value.isocalendar().week)

    return row


def recursive_forecast(
    model,
    history: Sequence[float],
    future_dates: Sequence[pd.Timestamp],
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> np.ndarray:
    history_buffer = list(history)
    preds: List[float] = []

    for current_date in future_dates:
        row = make_feature_row(history_buffer, current_date, lags, rolling_windows)
        X_row = pd.DataFrame([row])
        y_hat = float(model.predict(X_row)[0])
        preds.append(y_hat)
        history_buffer.append(y_hat)

    return np.asarray(preds)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    denom_mape = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom_mape)) * 100

    smape_denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_denom = np.where(smape_denom < 1e-8, 1e-8, smape_denom)
    smape = np.mean(np.abs(y_true - y_pred) / smape_denom) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
    }
    
## funcion para entrenar el modelo de suavizado exponencial

def train_exponential_smoothing(train_series: pd.Series, args: argparse.Namespace):
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels no esta instalado. Instala dependencias con: pip install -r requirements.txt"
        ) from exc

    # Convertir None strings a None reales
    trend = None if args.exsm_trend == "None" else args.exsm_trend
    seasonal = None if args.exsm_seasonal == "None" else args.exsm_seasonal

    model = ExponentialSmoothing(
        train_series.values,
        seasonal_periods=args.exsm_seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    return fitted


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
):
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError(
                "XGBoost no esta instalado. Instala dependencias con: pip install -r requirements.txt"
            ) from exc

        model = XGBRegressor(
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Modelo no soportado: {model_name}")


def save_predictions(path: Path, dates: pd.Series, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    out = pd.DataFrame(
        {
            "date": dates,
            "y_true": y_true,
            "y_pred": y_pred,
            "error": y_true - y_pred,
        }
    )
    out.to_csv(path, index=False)


def validate_window_requirements(
    train_size: int,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> None:
    min_history = max(max(lags), max(rolling_windows))
    if train_size <= min_history:
        raise ValueError(
            f"Train insuficiente. Se requieren mas de {min_history} filas para los lags/rolling seleccionados."
        )


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    test_path = Path(args.test_file)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(train_path, args.target)
    test_df = load_dataset(test_path, args.target)

    ml_models = [m for m in args.models if m != "exsm"]

    # Preparar datos supervisados solo si hay modelos ML.
    X_train: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    feature_names: List[str] = []
    initial_history: List[float] = []
    if ml_models:
        validate_window_requirements(len(train_df), args.lags, args.rolling_windows)
        X_train, y_train = build_supervised_matrix(
            train_df[args.target],
            train_df["date"],
            args.lags,
            args.rolling_windows,
        )
        feature_names = X_train.columns.tolist()
        initial_history = train_df[args.target].tolist()

    artifacts: List[ForecastArtifacts] = []
    model_feature_importances: Dict[str, Dict[str, float]] = {}
    y_true = test_df[args.target].to_numpy()

    for model_name in args.models:
        if model_name == "exsm":
            print(f"\nEntrenando Exponential Smoothing (seasonal_periods={args.exsm_seasonal_periods})...")
            fitted = train_exponential_smoothing(train_df[args.target], args)
            y_pred = np.asarray(fitted.forecast(steps=len(test_df)))
            model_feature_importances[model_name] = {}
            model_path = models_dir / "exsm_model.joblib"
            joblib.dump(fitted, model_path)
        else:
            assert X_train is not None and y_train is not None
            model = train_model(model_name, X_train, y_train, args)
            model_feature_importances[model_name] = extract_feature_importances(model, feature_names)
            y_pred = recursive_forecast(
                model=model,
                history=initial_history,
                future_dates=test_df["date"].tolist(),
                lags=args.lags,
                rolling_windows=args.rolling_windows,
            )
            model_path = models_dir / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)

        metrics = metric_bundle(y_true, y_pred)
        pred_path = output_dir / f"predictions_{model_name}.csv"
        save_predictions(pred_path, test_df["date"], y_true, y_pred)

        artifacts.append(
            ForecastArtifacts(
                model_name=model_name,
                predictions=pd.DataFrame({"date": test_df["date"], "pred": y_pred}),
                metrics=metrics,
                model_path=model_path,
            )
        )

    metrics_rows = []
    for item in artifacts:
        row = {"model": item.model_name}
        row.update(item.metrics)
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    metrics_path = output_dir / "metrics_comparison.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Features summary solo para modelos ML con importancias.
    if feature_names:
        ml_with_importances = [m for m in ml_models if model_feature_importances.get(m)]
        features_summary_rows = []
        for feature_name in feature_names:
            row = {
                "feature": feature_name,
                "feature_group": feature_group(feature_name),
            }
            for model_name in ml_with_importances:
                row[f"importance_{model_name}"] = model_feature_importances.get(model_name, {}).get(
                    feature_name, np.nan
                )
            features_summary_rows.append(row)

        features_summary_df = pd.DataFrame(features_summary_rows)
        features_summary_path = output_dir / "features_summary.csv"
        features_summary_df.to_csv(features_summary_path, index=False)

    best_model = metrics_df.iloc[0]["model"] if not metrics_df.empty else ""

    exsm_params: Dict = {}
    if "exsm" in args.models:
        exsm_params = {
            "seasonal_periods": args.exsm_seasonal_periods,
            "trend": args.exsm_trend,
            "seasonal": args.exsm_seasonal,
        }

    run_metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "models": args.models,
        "exsm_params": exsm_params,
        "lags": args.lags if ml_models else [],
        "rolling_windows": args.rolling_windows if ml_models else [],
        "features_used": feature_names,
        "best_model_by_rmse": best_model,
        "artifacts": [
            {
                "model": item.model_name,
                "model_path": str(item.model_path),
                "metrics": item.metrics,
            }
            for item in artifacts
        ],
    }
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print("\n=== Forecasting completado ===")
    print(f"Target: {args.target}")
    print(f"Modelos ejecutados: {', '.join(args.models)}")
    print(f"Mejor modelo (RMSE): {best_model}")
    print("\nMetricas:")
    print(metrics_df.to_string(index=False))
    print("\nArchivos generados:")
    print(f"- {metrics_path}")
    for model_name in args.models:
        print(f"- {output_dir / f'predictions_{model_name}.csv'}")
        suffix = "joblib"
        print(f"- {models_dir / f'{model_name}_model.{suffix}'}")
    print(f"- {metadata_path}")


if __name__ == "__main__":
    main()
