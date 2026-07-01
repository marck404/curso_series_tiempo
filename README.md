Bienvenidos a este curso de series de tiempo. El objetivo es estudiar y aprender sobre pronostico de series de tiempo y modelos usados en la industria para esta tarea.

El temario y las clases estan en desarrollo.

## Forecasting CLI

Se agrego un script ejecutable desde terminal para pronosticar `meantemp` con modelos de machine learning basados en rezagos.

### 1) Instalar dependencias

```bash
python -m pip install -r requirements.txt
```

### 2) Ejecutar entrenamiento + prediccion + evaluacion

```bash
python forecast_climate.py \
	--train-file datasets/DailyDelhiClimateTrain.csv \
	--test-file datasets/DailyDelhiClimateTest.csv \
	--models rf xgb \
	--output-dir results
```

### 3) Salidas esperadas

- `results/metrics_comparison.csv`: tabla de MAE, RMSE, MAPE y sMAPE por modelo.
- `results/predictions_rf.csv` y `results/predictions_xgb.csv`: predicciones y error diario.
- `results/models/rf_model.joblib` y `results/models/xgb_model.joblib`: modelos serializados.
- `results/run_metadata.json`: metadatos de la corrida y mejor modelo por RMSE.

### Opciones utiles

```bash
python forecast_climate.py --help
```

Permite configurar target, semilla, lista de lags, ventanas rolling y parametros principales de cada modelo.
