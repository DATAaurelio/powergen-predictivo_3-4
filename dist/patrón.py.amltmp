# Descarga de datos
datos = fetch_dataset(name='h2o_exog', raw=True)

# Preparación del dato
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')
datos = datos.set_index('fecha')
datos = datos.asfreq('MS')  # Convierte Timeseries a frecuencia especificada.
datos = datos.sort_index()
datos.head()

print(f'Número de filas con missing values: {datos.isnull().any(axis=1).mean()}')

# Verificar que un índice temporal está completo
fecha_inicio = datos.index.min()
fecha_fin = datos.index.max()
date_range_completo = pd.date_range(start=fecha_inicio, end=fecha_fin, freq=datos.index.freq)
print(f"Índice completo: {(datos.index == date_range_completo).all()}")

# Completar huecos en un índice temporal
datos.asfreq(freq='30min', fill_value=np.nan)

# Separación datos train-test
steps = 36
datos_train = datos[:-steps]
datos_test  = datos[-steps:]
print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")
fig, ax = plt.subplots(figsize=(6, 2.5))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
ax.legend();

# 6. Forecasting autorregresivo recursivo
# Crear y entrenar forecaster
forecaster = ForecasterRecursive(
    regressor=RandomForestRegressor(random_state=123),
    lags=10
)
forecaster.fit(y=datos_train['y'])
forecaster

# Predicciones
steps = 36
predicciones = forecaster.predict(steps=steps)
predicciones.head(5)

# Gráfico de predicciones vs valores reales
fig, ax = plt.subplots(figsize=(6, 2.5))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend()

# Error test
error_mse = mean_squared_error(
    y_true=datos_test['y'],
    y_pred=predicciones
)
print(f"Error de test (mse): {error_mse}")

# Búsqueda de hiperparámetros: grid search

forecaster = ForecasterRecursive(
    regressor=RandomForestRegressor(random_state=123),
    lags=12  # Este valor será remplazado en el grid search
)

# Particiones de entrenamiento y validación
cv = TimeSeriesFold(
    steps=36,
    initial_train_size=int(len(datos_train) * 0.5),
    refit=False,
    fixed_train_size=False,

)

# Valores candidatos de lags
lags_grid = [10, 20]

# Valores candidatos de hiperparámetros del regresor
param_grid = {
    'n_estimators': [100, 250],
    'max_depth': [3, 5, 10]
}

resultados_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=datos_train['y'],
    cv=cv,
    param_grid=param_grid,
    lags_grid=lags_grid,
    metric='mean_squared_error',
    return_best=True,
    n_jobs='auto',
    verbose=False
)

# Resultados de la búsqueda de hiperparámetros
resultados_grid

# Crear y entrenar forecaster con mejores hiperparámetros
regressor = RandomForestRegressor(n_estimators=250, max_depth=3, random_state=123)

forecaster = ForecasterRecursive(
    regressor=regressor,
    lags=20
)

forecaster.fit(y=datos_train['y'])

# Predicciones
predicciones = forecaster.predict(steps=steps)

# Gráfico de predicciones vs valores reales
fig, ax = plt.subplots(figsize=(6, 2.5))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend()

# Error de test
error_mse = mean_squared_error(
    y_true=datos_test['y'],
    y_pred=predicciones
)

print(f"Error de test (mse) {error_mse}")

# Backtesting
cv = TimeSeriesFold(
    steps=12 * 3,
    initial_train_size=len(datos) - 12 * 9,
    fixed_train_size=False,
    refit=True,
)

metrica, predicciones_backtest = backtesting_forecaster(
    forecaster=forecaster,
    y=datos['y'],
    cv=cv,
    metric='mean_squared_error',
    verbose=True
)
metrica

# Gráfico de predicciones de backtest vs valores reales
fig, ax = plt.subplots(figsize=(6, 2.5))
datos.loc[predicciones_backtest.index, 'y'].plot(ax=ax, label='test')
predicciones_backtest.plot(ax=ax, label='predicciones')
ax.legend()

# Importancia predictores
importancia = forecaster.get_feature_importances()
importancia.head(10)

# Matrices de entrenamiento utilizadas por el forecaster para entrenar el regresor

X_train, y_train = forecaster.create_train_X_y(y=datos_train['y'])

# Crear SHAP explainer (para modelos basados en árboles)
explainer = shap.TreeExplainer(forecaster.regressor)

# Se selecciona una muestra del 50% de los datos para acelerar el cálculo
rng = np.random.default_rng(seed=785412)
sample = rng.choice(X_train.index, size=int(len(X_train)*0.5), replace=False)
X_train_sample = X_train.loc[sample, :]
shap_values = explainer.shap_values(X_train_sample)

# Shap summary plot (top 10)
shap.initjs()
shap.summary_plot(shap_values, X_train_sample, max_display=10, show=False)
fig, ax = plt.gcf(), plt.gca()
ax.set_title("SHAP Summary plot")
ax.tick_params(labelsize=8)
fig.set_size_inches(6, 3.5)

#7. Forecasting con variables exógenas
# Descarga de datos
datos = fetch_dataset(name='h2o_exog', raw=True, verbose=False)

# Preparación del dato
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')
datos = datos.set_index('fecha')
datos = datos.asfreq('MS')
datos = datos.sort_index()

fig, ax = plt.subplots(figsize=(6, 2.5))
datos['y'].plot(ax=ax, label='y')
datos['exog_1'].plot(ax=ax, label='variable exógena')
ax.legend()

# Separación datos train-test
steps = 36
datos_train = datos[:-steps]
datos_test  = datos[-steps:]

print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

# Crear y entrenar forecaster
forecaster = ForecasterRecursive(
    regressor=RandomForestRegressor(random_state=123),
    lags=20
)

forecaster.fit(y=datos_train['y'], exog=datos_train['exog_1'])

forecaster

# Predicciones
predicciones = forecaster.predict(steps=steps, exog=datos_test['exog_1'])

# Gráfico predicciones vs valores reales

fig, ax = plt.subplots(figsize=(6, 2.5))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')

predicciones.plot(ax=ax, label='predicciones', color='orange')
ax.legend();

# Error test
error_mse = mean_squared_error(
                y_true = datos_test['y'],
                y_pred = predicciones
            )
print(f"Error de test (mse): {error_mse}")

importancia = forecaster.get_feature_importances()
importancia.head(10)

# 8. Predictores custom y window features
# Descarga de datos

datos = fetch_dataset(name='h2o_exog', raw=True, verbose=False)

# Preparación del dato
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')
datos = datos.set_index('fecha')
datos = datos.asfreq('MS')
datos = datos.sort_index()

# Separación datos train-test
steps = 36
datos_train = datos[:-steps]
datos_test  = datos[-steps:]

print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

# Window features
window_features = RollingFeatures(
    stats=['mean', 'std', 'min', 'max'],
    window_sizes=20
)

# Crear y entrenar forecaster

forecaster = ForecasterRecursive(
    regressor=RandomForestRegressor(random_state=123),
    lags=12,
    window_features=window_features,
)

forecaster.fit(y=datos_train['y'])

forecaster

# Matrices de entrenamiento

X_train, y_train = forecaster.create_train_X_y(y=datos_train['y'])

display(X_train.head(5))
display(y_train.head(5))

# Predicciones

steps = 36
predicciones = forecaster.predict(steps=steps)
# Gráfico predicciones vs valores reales

fig, ax = plt.subplots(figsize=(6, 2.5))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend()

# Error test

error_mse = mean_squared_error(
    y_true=datos_test['y'],
    y_pred=predicciones
)

print(f"Error de test (mse): {error_mse}")

# 9. Direct multi-step forecasting
# Crear forecaster

forecaster = ForecasterDirect(
    regressor=Ridge(random_state=123),
    transformer_y=StandardScaler(),
    steps=36,
    lags=8
)

forecaster

# Búsqueda de hiperparámetros

from skforecast.exceptions import LongTrainingWarning
warnings.simplefilter('ignore', category=LongTrainingWarning)

forecaster = ForecasterDirect(
    regressor=Ridge(random_state=123),
    transformer_y=StandardScaler(),
    steps=36,
    lags=8  # Este valor será remplazado en el grid search
)

cv = TimeSeriesFold(
    steps=36,
    initial_train_size=int(len(datos_train) * 0.5),
    fixed_train_size=False,
    refit=False,
)

param_grid = {'alpha': np.logspace(-5, 5, 10)}

lags_grid = [5, 12, 20]

resultados_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=datos_train['y'],
    cv=cv,
    param_grid=param_grid,
    lags_grid=lags_grid,
    metric='mean_squared_error',
    return_best=True,
    n_jobs='auto',
    verbose=False
)

# Resultados de la búsqueda de hiperparámetros
resultados_grid.head()

# Predicciones

predicciones = forecaster.predict()

# Gráfico predicciones vs valores reales

fig, ax = plt.subplots(figsize=(6, 2.5))
datos_train['y'].plot(ax=ax, label='train')
datos_test['y'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();

# 10. Intervalos de predicción

# Descarga de datos
# ==============================================================================
datos = fetch_dataset(name='h2o_exog', raw=True, verbose=False)

# Preparación del dato
# ==============================================================================
datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y-%m-%d')
datos = datos.set_index('fecha')
datos = datos.asfreq('MS')
datos = datos.sort_index()

# Separación datos train-test
# ==============================================================================
steps = 36
datos_train = datos[:-steps]
datos_test  = datos[-steps:]
print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

# Crear y entrenar forecaster
# ==============================================================================
forecaster = ForecasterRecursive(
    regressor=Ridge(alpha=0.1, random_state=765),
    lags=15
)

forecaster.fit(y=datos_train['y'])

# Intervalos de predicción
# ==============================================================================
predicciones = forecaster.predict_interval(
    steps=steps,
    interval=[1, 99],
    n_boot=500
)

predicciones.head(5)

# Error de predicción
# ==============================================================================
error_mse = mean_squared_error(
    y_true=datos_test['y'],
    y_pred=predicciones['pred']
)
print(f"Error de test (mse): {error_mse}")

# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
datos_test['y'].plot(ax=ax, label='test')
predicciones['pred'].plot(ax=ax, label='predicciones')
ax.fill_between(
    predicciones.index,
    predicciones['lower_bound'],
    predicciones['upper_bound'],
    color='red',
    alpha=0.2
)
ax.legend(loc='upper left')

# Backtest con intervalos de predicción
# ==============================================================================

forecaster = ForecasterRecursive(
    regressor=Ridge(alpha=0.1, random_state=765),
    lags=15
)
cv = TimeSeriesFold(
    steps=36,
    initial_train_size=len(datos) - 12 * 9,
    fixed_train_size=False,
    refit=True,
)

metrica, predicciones = backtesting_forecaster(
    forecaster=forecaster,
    y=datos['y'],
    cv=cv,
    metric='mean_squared_error',
    interval=[1, 99],
    n_boot=100,
    n_jobs='auto',
    verbose=True
)

display(metrica)

# Gráfico
# ==============================================================================
fig, ax = plt.subplots(figsize=(6, 2.5))
datos.loc[predicciones.index, 'y'].plot(ax=ax, label='test')
predicciones['pred'].plot(ax=ax, label='predicciones')
ax.fill_between(
    predicciones.index,
    predicciones['lower_bound'],
    predicciones['upper_bound'],
    color='red',
    alpha=0.2
)
ax.legend()

# Cobertura del intervalo predicho
# ==============================================================================
dentro_intervalo = np.where(
    (datos.loc[predicciones.index, 'y'] >= predicciones['lower_bound']) &
    (datos.loc[predicciones.index, 'y'] <= predicciones['upper_bound']),
    True,
    False
)

cobertura = dentro_intervalo.mean()
print(f"Cobertura del intervalo predicho: {round(100*cobertura, 2)} %")

# 11. Métrica custom
# Métrica custom
# ==============================================================================
def custom_metric(y_true, y_pred):
    '''
    Calcular el mean_absolute_error utilizando únicamente las predicciones de
    los últimos 3 meses del año.
    '''
    mask = y_true.index.month.isin([10, 11, 12])
    metric = mean_absolute_error(y_true[mask], y_pred[mask])

    return metric


# Backtesting
# ==============================================================================
metrica, predicciones_backtest = backtesting_forecaster(
    forecaster=forecaster,
    y=datos['y'],
    cv=cv,
    metric=custom_metric,
    n_jobs='auto',
    verbose=True
)

metrica

# 12. Guardar y cargar modelos
# Crear forecaster
# ==============================================================================
forecaster = ForecasterRecursive(RandomForestRegressor(random_state=123), lags=3)
forecaster.fit(y=datos['y'])
forecaster.predict(steps=3)
# Guardar modelo
# ==============================================================================
save_forecaster(forecaster, file_name='forecaster.joblib', verbose=False)
# Cargar modelo
# ==============================================================================
forecaster_cargado = load_forecaster('forecaster.joblib')
# Predicciones
# ==============================================================================
forecaster_cargado.predict(steps=3)
# Crear y entrenar forecaster
# ==============================================================================
forecaster = ForecasterRecursive(
    regressor=RandomForestRegressor(random_state=123),
    lags=6
)

forecaster.fit(y=datos_train['y'])
# Predecir con last_window
# ==============================================================================
last_window = datos_test['y'][-6:]
forecaster.predict(last_window=last_window, steps=4)

# Arima
import numpy as np
import pandas as pd
# ! pip install statsmodels
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('../../data/ARIMA/dataset.txt')

print(df.describe())
print(df.head())

from statsmodels.tsa.stattools import adfuller
from numpy import log

result = adfuller(df.value.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Diferienciamos
result = adfuller(df.value.diff().dropna())
print('ADF 1st Order Differencing:: %f' % result[0])
print('p-value: %f' % result[1])

result = adfuller(df.value.diff().diff().dropna())
print('ADF 2nd Order Differencing: %f' % result[0])
print('p-value: %f' % result[1])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})


# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1], lags=50, alpha=0.05)

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1], lags=50, alpha=0.05)

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1], lags=50, alpha=0.05)

plt.show()

# PACF
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

plot_pacf(df.value.diff().dropna(), lags=20, alpha=0.05)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

plot_acf(df.value.diff().diff().dropna(), lags=20, alpha=0.05)

plt.show()

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tabulate import tabulate

# ARIMA Model
model = ARIMA(df.value, order=(1,2,2))
# model = SARIMAX(df.value, order=(1,2,2))
model_fit = model.fit()
print(model_fit.summary())

# errores residuales
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

from statsmodels.graphics.tsaplots import plot_predict

# Actual vs Fitted
plot_predict(model_fit, 2, 200, dynamic=False, ax=df.plot())

plt.show()

from statsmodels.tsa.stattools import acf

# Crear entrenamiento y prueba
train_size = 190
train = df.value[:train_size+1]
test = df.value[train_size:]

from tabulate import tabulate

# Build del modelo
model = ARIMA(train, order=(1, 2, 2))
model_fit = model.fit()
print(model_fit.summary())

# Pronóstico
steps = len(df) - train_size
fc_series = model_fit.forecast(steps=steps, alpha=0.05)  # 95% conf
conf = model_fit.get_forecast(steps=steps).conf_int(alpha=0.05)

# pandas series
lower_series = conf['lower value']
upper_series = conf['upper value']

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast', color='red')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Build Model
# model = ARIMA(train, order=(1, 2, 2))
model = ARIMA(train, order=(2, 2, 1))
model_fit = model.fit()
print(model_fit.summary())

# Pronóstico
fc_series = model_fit.forecast(steps, alpha=0.05)  # 95% conf
conf = model_fit.get_forecast(steps).conf_int(alpha=0.05)

# pandas series
lower_series = conf['lower value']  # pd.Series(conf[:, 0], index=test.index)
upper_series = conf['upper value']  # pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast', color='red')
plt.fill_between(lower_series.index, lower_series,
                 upper_series,  color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

from sklearn.metrics import mean_squared_error

data = df.value

# Parámetros de la validación cruzada
initial_train_size = int(len(data) * 0.8)  # 80% para entrenamiento inicial
steps = 10  # Número de pasos a predecir en cada iteración
n_backtests = int((len(data) - initial_train_size) / steps)

# Listas para almacenar resultados
predictions = []
actuals = []

# Validación cruzada fuera del tiempo
for i in range(n_backtests):
    # Dividir datos en entrenamiento y prueba
    train_data = data[:initial_train_size + i * steps]
    test_data = data[initial_train_size + i * steps: initial_train_size + (i + 1) * steps]

    # Entrenar el modelo ARIMA
    model = ARIMA(train_data, order=(1, 1, 1))
    model_fit = model.fit()

    # Hacer predicciones
    forecast = model_fit.forecast(steps=steps)
    predictions.extend(forecast)
    actuals.extend(test_data)

# Evaluar el rendimiento
mse = mean_squared_error(actuals, predictions)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

# Mostrar resultados
results = pd.DataFrame({"Actual": actuals, "Predicted": predictions})

results.head()

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast.to_numpy()[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast.to_numpy()[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(forecast-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc_series, test.values)

# ! pip install "numpy<2"
# ! pip install pmdarima

import pandas as pd
import numpy as np

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

import pmdarima as pm

model = pm.auto_arima(y=df.value,
                      start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3,  # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True
                      )

print(model.summary())

model.plot_diagnostics(figsize=(10,8))
plt.show()

# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True) # forecast, confidence_intervals
index_of_fc = np.arange(len(df.value), len(df.value)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.value)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Usage")
plt.show()

# 14. Modelo de SARIMA en Python** <a class="anchor
data = pd.read_csv('../../data/ARIMA/dataset.txt', parse_dates=['date'], index_col='date')

# Plot
fig, axes = plt.subplots(2, 1, figsize=(16,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)


# Seasonal Differencing
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('Drug Sales - Time Series Dataset', fontsize=16)
plt.show()

import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(data, start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()

# Forecast
n_periods = 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(data.index[-1], periods = n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(data)
plt.plot(fitted_series, color='red')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMA - Final Forecast of Drug Sales - Time Series Dataset")
plt.show()

# 15. Modelo SARIMAX con variables exogéneas
# Calcular índice estacional
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# componente estacional multiplicativo
result_mul = seasonal_decompose(data['value'][-36:],   # 3 years
                                model='multiplicative', 
                                extrapolate_trend='freq')

seasonal_index = result_mul.seasonal[-12:].to_frame()
seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

# merge con los datos base
data['month'] = data.index.month
df = pd.merge(data, seasonal_index, how='left', on='month')
df.columns = ['value', 'month', 'seasonal_index']
df.index = data.index  # reassign the index
print(tabulate(df, headers='keys', tablefmt='psql'))

import pmdarima as pm

# SARIMAX Model
sxmodel = pm.auto_arima(df[['value']], exogenous=df[['seasonal_index']],
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

sxmodel.summary()

# Forecast
n_periods = 24
fitted, confint = sxmodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(data.index[-1], periods = n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(data['value'])
plt.plot(fitted_series, color='red')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("SARIMAX - Final Forecast of Drug Sales - Time Series Dataset")
plt.show()

# Series temporales multivariadas con Auto ARIMA
from tabulate import tabulate

def tabl_prt(to_primt):
    print(tabulate(to_primt, headers='keys',tablefmt='psql'),'\n')

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('../../data/ARIMA/energy_consumption.csv')

tabl_prt(df.describe(include='all'))

tabl_prt(df.head())
tabl_prt(df.tail())

# Convertir la columna de marca de tiempo

df['timeStamp']=pd.to_datetime(df['timeStamp'])

# Trazar la columna de demanda
import plotly.express as px

fig = px.line(df, x='timeStamp', y='demand', title='Energy Consumption')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(step="all")
        ])
    )
)
fig.show()

el_df=df.set_index('timeStamp')

el_df.plot(subplots=True)

print ("\nMissing values :  ", df.isnull().any())

df['demand']=df['demand'].ffill()

df['temp']=df['temp'].ffill()

df['temp']=df['precip'].ffill()

print ("\nMissing values :  ", df.isnull().any())

el_df.resample('ME').mean()

el_df.resample('ME').mean().plot(subplots=True)

# Guardar el conjunto de datos resmuestrado

final_df=el_df.resample('ME').mean()

import pmdarima as pm

model = pm.auto_arima(final_df['demand'],
                      m=12, seasonal=True,
                      start_p=0, start_q=0, max_order=4, test='adf', error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True, trace=True)

train = final_df[(final_df.index.get_level_values(0) >= '2012-01-31')
                 & (final_df.index.get_level_values(0) <= '2014-04-30')]

train.tail()

test = final_df[(final_df.index.get_level_values(0) > '2014-04-30')]

test

model.fit(train['demand'])

forecast = model.predict(n_periods=8, return_conf_int=True)

# Predicciones
forecast_df = pd.DataFrame(forecast[0], index=test.index, columns=['Prediction'])

forecast_df

# Graficando

import matplotlib.pyplot as plt
pd.concat([final_df['demand'],forecast_df],axis=1).plot()

forecast1=model.predict(n_periods=8, return_conf_int=True)
forecast_range=pd.date_range(start='2014-05-31', periods=8,freq='ME')

forecast1_df = pd.DataFrame(forecast1[0], index=forecast_range, columns=['Prediction'])

forecast1_df

# Graficando las predicciones
pd.concat([final_df['demand'],forecast1_df],axis=1).plot()


































