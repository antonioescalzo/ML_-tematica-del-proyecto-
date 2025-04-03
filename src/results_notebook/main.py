# Fechas y configuración de warnings
import datetime
import warnings
warnings.filterwarnings("ignore")  

# Visualización
import matplotlib.pyplot as plt  
import numpy as np              
# Machine Learning y optimización
import optuna  
from sklearn.metrics import mean_absolute_error, mean_squared_error  
from sklearn.model_selection import train_test_split 
# Modelos de series temporales
from statsmodels.tsa.arima.model import ARIMA  
from pmdarima import auto_arima  
from prophet import Prophet  
from xgboost import XGBRegressor  

# Fuentes de datos financieros
import yfinance as yf  
import pandas_datareader as pdr  
# Manipulación de datos
import pandas as pd  

#USAMOS PLT PARA ESTILIAR MUCHO MAS LAS GFRÁFICAS UN TOQUE MAS "PROFESIONAL"
plt.style.use('ggplot')
#DESCARGAMOS DATOS HISTÓRICOS DE APPLE (AAPL) 
stock_data = yf.download("AAPL")

# Si las columnas tienen MultiIndex, simplificarlas
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

num_days_pred=30
#USAMOS PLT PARA ESTILIAR MUCHO MAS LAS GFRÁFICAS UN TOQUE MAS "PROFESIONAL"
plt.style.use('ggplot')
#DESCARGAMOS DATOS HISTÓRICOS DE APPLE (AAPL) 
stock_data = yf.download("AAPL")

# Si las columnas tienen MultiIndex, simplificarlas
if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)

num_days_pred=30
print(stock_data.columns)
stock_data.info()
#ESCOJO ÚNICAMENTE LA COLUMNA DE VALORES DE CIERRE
stock_data.drop(columns=['Open', 'High','Low', 'Volume'], inplace=True)
#FUNCIÓN PARA CALCULAR EL PORCENTAJE DE ERROR MEDIO ABSOLUTO (MAE%)

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcula el ERROR  PORCENTUAL ABSOLUTO MEDIO (MAE) a partir de valores reales(y_true)
    y los valores predecidos(y_pred) """
    y_true, y_pred= np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))*100


# Creamos variables rezagadas (lags) de la columna 'Close' para capturar patrones temporales en el modelo

def add_Lags(df, num_days_pred=30):  # Ojo al return al final
    target = 'Close'
    df['lag1'] = df[target].shift(num_days_pred)
    df['lag2'] = df[target].shift(num_days_pred*2)
    df['lag3'] = df[target].shift(num_days_pred*3)
    df['lag4'] = df[target].shift(num_days_pred*4)
    df['lag5'] = df[target].shift(num_days_pred*5)
    df['lag6'] = df[target].shift(num_days_pred*6)
    df['lag7'] = df[target].shift(num_days_pred*7)
    df['lag8'] = df[target].shift(num_days_pred*8)
    df['lag9'] = df[target].shift(num_days_pred*9)
    df['lag10'] = df[target].shift(num_days_pred*10)
    df['lag11'] = df[target].shift(num_days_pred*11)
    df['lag12'] = df[target].shift(num_days_pred*12)
    return df  # ← Esto es fundamental
# Extraemos características temporales del índice para enriquecer el modelo con contexto estacional y periódico

def create_features(df):
    """
    Creamos características de tiempo basadas en el índice de la serie temporal.
    Esto nos permite incluir información como el mes, día, semana, etc.,
    que puede ser muy útil para detectar patrones estacionales.
    """
    df = df.copy()
    df['hour'] = df.index.hour                # Hora del día
    df['dayofweek'] = df.index.dayofweek      # Día de la semana (0=lunes)
    df['quarter'] = df.index.quarter          # Trimestre del año
    df['month'] = df.index.month              # Mes
    df['year'] = df.index.year                # Año
    df['dayofyear'] = df.index.dayofyear      # Día del año
    df['dayofmonth'] = df.index.day           # Día del mes
    df['weekofyear'] = df.index.isocalendar().week  # Semana del año (ISO)
    return df
#Hacemos una copia del modelo 

df_xgb = stock_data.copy()

# Preparamos los datos para el modelo XGBoost: aplicamos transformación temporal y variables
def xgboostmodel(df_xgb, add_lags, create_features, num_days_pred=30):
    df_xgb = create_features(df_xgb)
    df_xgb = add_lags(df_xgb, num_days_pred=num_days_pred)  # pasar también el parámetro
    df_xgb.dropna(inplace=True)  # Opcional: elimina filas con NaN por los shifts
    X = df_xgb.drop(columns='Close')
    y = df_xgb['Close']
    return X, y

X, y = xgboostmodel(df_xgb, add_Lags, create_features, num_days_pred=30)
#AHORA VAMOS A DAR PASO A OPTUNA, UNA LIBRERÍA DE OPTIMIZACIÓN AUTOMÁTICA DE HIPERPARÁMETROS BASADA EN LOS MEJORES ENSAYOS
# Define objective function for Optuna
def objective(trial):
    # Definimos los hiperpárametros que se van a optimizar con OPTUNA

    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'verbosity': 0,
        #'tree_method': 'gpu_hist',
    }

    # Esto inicializa un modelo XGBoost para regresión.
    xgb = XGBRegressor(**param)
    
    # Entrena el modelo con los datos de entrenamiento
    xgb.fit(X_train, y_train)
    
    # Usa el modelo entrenado para predecir valores sobre el conjunto de prueba
    y_pred = xgb.predict(X_test)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rmse
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 #Crear y ejecutar estudio de Optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Imprimir mejores resultados encontrados
print("Best trial:")
best_trial = study.best_trial
print(f"  Value: {best_trial.value}")
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Entrenar el modelo final con los mejores hiperparámetros
best_params = best_trial.params
xgb_best = XGBRegressor(**best_params)
xgb_best.fit(X_train, y_train)

# Predicción y evaluación final
y_pred_test = xgb_best.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print("Test RMSE:", rmse_test)
# Evaluamos el rendimiento del modelo XGBoost en el conjunto de test

y_pred_test_xgb = xgb_best.predict(X_test)
xgb_loss = mean_absolute_percentage_error(y_test, y_pred_test_xgb) 
print(f"ERROR PERCENT = { mean_absolute_percentage_error(y_test, y_pred_test_xgb) }% ")
# Visualizamos los valores reales frente a los valores predichos por el modelo XGBoost

plt.figure(figsize=(10, 6))
plt.scatter(X_test.index, y_test, color='blue', label='Actual')
plt.scatter(X_test.index, y_pred_test_xgb , color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
# Hacemos un zoom para visualizar mejor los primeros 30 valores predichos
plt.figure(figsize=(10, 6))
plt.scatter(X_test.index[:30], y_test[:30], color='blue', label='Actual')
plt.scatter(X_test.index[:30], y_pred_test_xgb[:30] , color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
# Visualizamos las 20 características más importantes según el modelo XGBoost


from xgboost import plot_importance

# Plot feature importance
plt.figure(figsize=(10, 6))
plot_importance(xgb_best, max_num_features=20)  # Adjust max_num_features as needed
plt.title("Feature Importance")
plt.show()

# Definimos el rango de fechas para hacer predicciones futuras


start = df_xgb.index.max()
end = start + pd.Timedelta(days=num_days_pred)
prediction_xgb = pd.DataFrame(future_w_features['pred'])
prediction_xgb
# Preparamos los datos para Prophet: dividimos en entrenamiento y test, y renombramos columnas a 'ds' y 'y'.


df_prophet = stock_data.copy()
split_date = df_prophet.index[int(len(df_prophet) * 0.8)]
train = df_prophet.loc[df_prophet.index <= split_date].copy()
test = df_prophet.loc[df_prophet.index > split_date].copy()
train_prophet = train.reset_index() \
    .rename(columns={'Date':'ds',
                     'Close':'y'})
train_prophet 
prophet = Prophet()
prophet.fit(train_prophet)
# Formateamos los datos para usarlos en Prophet renombrando columnas y generamos predicciones con el modelo.

test_prophet = test.reset_index() \
    .rename(columns={'Date':'ds',
                     'Close':'y'})
test_predict = prophet.predict(test_prophet)

# Calculamos el error porcentual del modelo Prophet usando MAPE entre valores reales y predichos.

porphet_loss = mean_absolute_percentage_error(test['Close'],test_predict['yhat'] )
print(f"ERROR PERCENT = { mean_absolute_percentage_error(test['Close'],test_predict['yhat'] ) }% ")
# Asegurarse de que el índice de test sea tipo fecha
test.index = pd.to_datetime(test.index)
test_predict.index = test.index  # Sincroniza el índice con test (si no lo tiene)

# Graficar con fechas reales en eje X
plt.figure(figsize=(10, 6))
plt.scatter(test.index, test['Close'], color='blue', label='Actual')
plt.scatter(test.index, test_predict['yhat'], color='red', label='Predicted')
plt.title('Actual vs Predicted Values')
plt.xlabel('Time')
plt.ylabel('Value')
plt.gcf().autofmt_xdate()  # Rotar fechas automáticamente
plt.legend()
plt.show()
prophet.plot_components(test_predict)
# Entrenamiento del modelo Prophet con los datos históricos formateados
prophet_data = df_prophet.reset_index() \
    .rename(columns={'Date': 'ds', 'Close': 'y'})
prophet = Prophet()
prophet.fit(prophet_data)
# Generamos predicciones futuras con Prophet y reindexamos el resultado por fecha
future = prophet.make_future_dataframe(periods=num_days_pred, freq='d', include_history=False)
forecast = prophet.predict(future) 
forecast_prophet = forecast[['ds', 'yhat']]
forecast_prophet.index = forecast_prophet.pop('ds')
forecast_prophet
df_arima = stock_data.copy()
split_date = df_prophet.index[int(len(df_arima) * 0.8)]
train_arima = df_arima.loc[df_arima.index <= split_date].copy()
test_arima = df_arima.loc[df_arima.index > split_date].copy()
# Buscamos automáticamente los mejores parámetros para el modelo ARIMA y los guardamos
stepwise_fit = auto_arima(train_arima['Close'], trace=True, suppress_warnings=True)
best_order = stepwise_fit.get_params()['order']
# Entrenamos el modelo ARIMA con los mejores parámetros encontrados
arima = ARIMA(train_arima['Close'], order=best_order)
arima = arima.fit()
# Definimos el rango de fechas para realizar predicciones con el modelo ARIMA
start = len(train_arima)
end = len(test_arima) + len(train_arima)
# Realizamos predicción con el modelo ARIMA y visualizamos los resultados
pred_arima = arima.predict(start=start, end=end-1)
pred_arima.index = test_arima.index
pred_arima.plot()
# Calculamos el error porcentual del modelo ARIMA usando MAPE
arima_loss = mean_absolute_percentage_error(test_arima['Close'], pred_arima)
print(f"ERROR PERCENT = { arima_loss }% ")
# Buscamos los mejores parámetros y entrenamos el modelo ARIMA con la serie completa
stepwise_fit = auto_arima(df_arima['Close'], trace=True, suppress_warnings=True)
best_order = stepwise_fit.get_params()['order']

arima = ARIMA(df_arima['Close'], order=best_order)
arima = arima.fit()
start = len(df_arima)
end = len(df_arima) + num_days_pred
arima_forecast = arima.predict(start=start,end=end)
df_arima.index = pd.to_datetime(df_arima.index)
# Asegúrate que el índice es datetime
df_arima.index = pd.to_datetime(df_arima.index)

# Obtenemos fecha final
start_date = df_arima.index.max()
end_date = start_date + pd.Timedelta(days=num_days_pred)

# Creamos fechas para el forecast
forecast_index = pd.date_range(start=start_date + pd.Timedelta(days=1), end=end_date, freq='1d')

# Predecimos
arima_forecast = arima.predict(start=len(df_arima), end=len(df_arima) + num_days_pred - 1)

# Le damos ese índice de fechas
arima_forecast.index = forecast_index# Graficamos las predicciones de Arima, XGBoost y Prophet para los próximos días
fig, ax  = plt.subplots(figsize=(10, 5))
arima_forecast.plot(color='orange', ax=ax)
prediction_xgb.plot(color='blue', ax=ax)
forecast_prophet.plot(color='red', ax=ax)
plt.legend(['Arima', 'XGB', 'Prophet'])
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title(f"Price Prediction in the Next {num_days_pred} Days")
# Mostramos la precisión (accuracy) de cada modelo restando el MAPE al 100%
print(f"XGB Acc : {100 - xgb_loss} \nArima Acc : {100 - arima_loss} \nProphet Acc : {100 - porphet_loss}")