# ML_-tematica-del-proyecto-
# Predicción de Series Temporales Financieras con Modelos de Machine Learning

## Problema

Este proyecto busca desarrollar modelos de machine learning para predecir precios de acciones basados en series temporales financieras. El objetivo es evaluar y comparar distintos modelos aplicados a datos históricos del mercado bursátil.

## Dataset

Se utiliza un dataset público descargado mediante la librería `yfinance`, que proporciona acceso a datos financieros históricos. En este caso se usan los datos de la acción de Apple Inc. (AAPL), incluyendo precios de apertura, cierre, máximos, mínimos y volumen.

- Origen: Yahoo Finance
- Acceso: El dataset se puede obtener con `yf.download("AAPL")` usando la librería `yfinance`.

## Solución Adoptada

El proyecto explora y compara distintos enfoques para la predicción de precios bursátiles a corto plazo, incluyendo:

- Modelos estadísticos como ARIMA y Auto-ARIMA
- Modelos basados en aprendizaje automático como XGBoost
- Modelos con componentes estacionales como Prophet
- Optimización de hiperparámetros con Optuna
- Evaluación mediante métricas MAE (Mean Absolute Error) y RMSE (Root Mean Squared Error)

Se busca predecir el comportamiento del precio de la acción a futuro, típicamente para los próximos 30 días.

## Estructura del Repositorio

Proyecto_Prediccion_Series/
│
├── README.md
├── SRC/
│   ├── data_sample/
│   │   └── dataset.csv               # Datos descargados o generados
│   ├── models/
│   │   └── modelo_entrenado.joblib   # Modelo entrenado guardado
│   ├── results_notebook/
│   │   └── 0-Guia_Proyecto_ML.ipynb  # Notebook con el desarrollo del proyecto
│   └── utils/
│       └── requirements.txt          # Lista de librerías necesarias


## Requisitos

Instalación de dependencias necesarias:
pandas
numpy
yfinance
ta
scikit-learn
xgboost
prophet
matplotlib
seaborn
textblob
transformers
tensorflow
keras-tuner
joblib
numba
requests
beautifulsoup4
lxml

## Notas

El proyecto puede ampliarse para predecir otras acciones o activos financieros. También es posible ajustar el horizonte de predicción y realizar análisis comparativos entre diferentes modelos.










