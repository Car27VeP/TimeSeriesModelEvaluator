# %% [markdown]
"""
# Descripción del proyecto

La compañía Sweet Lift Taxi ha recopilado datos históricos sobre pedidos de taxis en los aeropuertos. Para atraer a más conductores durante las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. Construye un modelo para dicha predicción.

La métrica RECM en el conjunto de prueba no debe ser superior a 48.

## Instrucciones del proyecto.

1. Descarga los datos y haz el remuestreo por una hora.
2. Analiza los datos
3. Entrena diferentes modelos con diferentes hiperparámetros. La muestra de prueba debe ser el 10% del conjunto de datos inicial.
4. Prueba los datos usando la muestra de prueba y proporciona una conclusión.

## Descripción de los datos

Los datos se almacenan en el archivo `taxi.csv`. 	
El número de pedidos está en la columna `num_orders`.
"""
# %% [markdown]
## 1. Preparacion
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# %% [markdown]
### Carga de los datos.
# %% 
df_taxi = pd.read_csv(r'../../../datasets/taxi.csv')
#num_orders = pd.read_csv(r'../../../datasets/num_orders.csv')
# %% [markdown]
## 2. Análisis
# %% [markdown]
### Observación general de los datos.
# %%
df_taxi.head()
# %%
df_taxi.info()
# %% [markdown]
"""
Obervamos que los datos se cargaron correctamente. 
El DataFrame 26 406 filas por 2 columnas, del cual no hay ningún registro nulo.
el tipo de valores que tiene nuestro dataframe es object e int64. Dado que tenemos una columa llamada datetime,
cambiar esta columna al formato correspondiente. También observamos el número de viajes registrados es
cada 10 minutos. 
"""
# %% [markdown]
### Obervación estadística de los datos.
# %%
df_taxi.describe()
# %%
df_taxi.plot(kind='box',title='Distribución de los viajes.')
plt.show()
# %% [markdown]
"""
En la descripción estadística observamos que
los datos se encuentran ligeramente sesgados a la derecha; 
sin embargo, no es un problema para seguir adelante. El número total de viajes fue de 26,496, 
hubo registros con cero viajes, 
y la cantidad máxima de viajes en una hora y día específicos fue de 119.
"""
# %% [markdown]
### Cambiar formato fecha y hacer remuestreo por una hora.
# %%
df_taxi['datetime'] = pd.to_datetime(df_taxi['datetime'])
df_taxi = df_taxi.set_index('datetime')
df_taxi = df_taxi.resample('1D').sum()
df = df_taxi.copy()
# %%
df['rolling_mean'] = df['num_orders'].rolling(5).mean()
df.plot(title="Viajes de marzo a agosto del 2018.")
plt.show()
# %% [markdown]
"""
En la gráfica podemos obsservar la cantoidad de viajes regitrados
de los días de marzo a agosto. Además suavizamos con promedio móvil la gráfica.
"""
# %%
## Tendencia, estacionalidad y residuales.
# %%
decomposed = seasonal_decompose(df['num_orders'])

plt.figure(figsize=(14, 20))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca())
plt.title('Tendencia')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca())
plt.title('Estacionalidad')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca())
plt.title('Residuales')
# %% [markdown]
### Hacer los datos mas estacionarios.
# %%
data = df[['num_orders']]
data -= data.shift()

data['mean'] = data['num_orders'].rolling(15).mean()
data['std'] = data['num_orders'].rolling(15).std()

data.plot(title='Datos más estacionarios.')
plt.show()
# %% [markdown]
## 3. Formación
# %%
data = data.drop(["mean","std"],axis=1)
train, test = train_test_split(data, test_size=0.1, shuffle=False)
print(train.shape)
print(test.shape)
# %% [markdown]
## Prueba
# %% [markdown]
### Pruebas de cordura.
# %% [markdown]
### Método 1.
# %%
print(f"Viajes medios al día: {data['num_orders'].median()}")
pred_median = np.ones(test.shape) * data['num_orders'].median()
print('EAM:', mae(test,pred_median))
# %% [markdown]
### Método 2.add()
print(f"Viajes medios al día: {data['num_orders'].median()}")
pred_previous = test.shift()
pred_previous.iloc[0] = train.iloc[-1]
print('EAM:', mae(test,pred_previous))
# %% [markdown]
"""
Nuestras prueba de cordura tuvieron un error demasiado grande, no les fue nada bien. 
En los siguientes modelos haremos que este valor sea menor.
"""
# %% [markdown]
### Creación de características para los modelos.
# %% 
def make_features(data, max_lag, rolling_mean_size):
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['dayofweek'] = data.index.dayofweek

    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['num_orders'].shift(lag)

    data['rolling_mean'] = (
        data['num_orders'].shift().rolling(rolling_mean_size).mean()
    )

make_features(data,4,4)
# data_features = data.drop('num_orders',axis=1)
# data_target = data['num_orders']
train, test = train_test_split(data, 
                               test_size=0.1, 
                               shuffle=False, 
                               random_state=1234)
train = train.dropna()


features_train = train.drop(['num_orders'], axis=1)
target_train = train['num_orders']
features_test = test.drop(['num_orders'], axis=1)
target_test = test['num_orders']
# %% [markdown]
### Regresión Logistica.
# %%
model = LogisticRegression(random_state=1234)
model.fit(features_train,target_train)
pred_train = model.predict(features_train)
pred_test = model.predict(features_test)

print(
    'EAM para el conjunto de entrenamiento:', 
    mae(target_train, pred_train)
)
print('EAM para el conjunto de prueba:', 
      mae(target_test, pred_test))

# %%
pred_test = pd.Series(pred_test,index=target_test.index)
train_with_predictions = pd.concat([target_train,pred_test])

train_predictions_rolling = train_with_predictions.rolling(15).mean()
df_rolling = data['num_orders'].rolling(15).mean()

plt.title("Promedio movil predicción.")
plt.plot(df_rolling, 
         label="Datos originales")
plt.plot(train_predictions_rolling.loc[test.index], 
         label="Predicción")
plt.legend()
plt.show()
# %%
# %% [markdown]
### Árboles de Decisión
# %%
mae_list = []
pred_test_list = []

for x in range(30):
    model = DecisionTreeRegressor(max_depth=x+1,
                                  random_state=1234)
    model.fit(features_train,target_train)
    
    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
      
    mae_list.append(mae(target_test, pred_test))
    pred_test_list.append(np.array(pred_test))
    
mae_list = np.array(mae_list)
pred_test_list = np.array(pred_test_list)
# %%
plt.plot(mae_list)
plt.plot(np.argmin(mae_list),np.min(mae_list), 
         marker='*', markersize=15,
         label=f"Mejor MAE: {np.min(mae_list):.2f}")
plt.title("MAE Árboles con 30 iteraciones")
plt.legend()
plt.show()
# %%
pred_test = pd.Series(pred_test_list[np.argmin(mae_list)-1],index=target_test.index)
train_with_predictions = pd.concat([target_train,pred_test])

train_predictions_rolling = train_with_predictions.rolling(15).mean()
df_rolling = data['num_orders'].rolling(15).mean()

plt.title("Promedio movil predicción con Árboles.")
plt.plot(df_rolling, 
         label="Datos originales")
plt.plot(train_predictions_rolling.loc[test.index], 
         label="Predicción")
plt.legend()
plt.show()
# %%
### Random Forest
# %%
mae_list = []
pred_test_list = []
for x in range(30):
    model = RandomForestRegressor(random_state=12345,
                                  criterion='absolute_error',
                                  max_depth=x+1)
    model.fit(features_train,target_train)
    
    pred_test = model.predict(features_test)
    
    mae_list.append(mae(target_test, pred_test))
    pred_test_list.append(np.array(pred_test))
    
mae_list = np.array(mae_list)
pred_test_list = np.array(pred_test_list)
# %%
plt.plot(mae_list)
plt.plot(np.argmin(mae_list),np.min(mae_list), 
         marker='*', markersize=15,
         label=f"Mejor MAE: {np.min(mae_list):.2f}")
plt.title("MAE Random Forest con 30 iteraciones")
plt.legend()
plt.show()
# %%
pred_test = pd.Series(pred_test_list[np.argmin(mae_list)-1],index=target_test.index)
train_with_predictions = pd.concat([target_train,pred_test])

train_predictions_rolling = train_with_predictions.rolling(15).mean()
df_rolling = data['num_orders'].rolling(15).mean()

plt.title("Promedio movil predicción con Random Forest.")
plt.plot(df_rolling, 
         label="Datos originales")
plt.plot(train_predictions_rolling.loc[test.index], 
         label="Predicción")
plt.legend()
plt.show()
# %% CatBoost
model = CatBoostRegressor()
parameters = {'depth' : [6,8,10],
              'learning_rate' : [0.01, 0.05, 0.1],
              'iterations'    : [30, 50, 100]
              }

grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1)
grid.fit(features_train, target_train)
# %%
print(f"Los mejores parametros para CatBoost: {grid.best_params_}")
# %%
model = CatBoostRegressor(**grid.best_params_)
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"\nMAE Catboost: {mae(target_test,pred_test)}")
# %%
pred_test = pd.Series(pred_test,index=target_test.index)
train_with_predictions = pd.concat([target_train,pred_test])

train_predictions_rolling = train_with_predictions.rolling(15).mean()
df_rolling = data['num_orders'].rolling(15).mean()

plt.title("Promedio movil predicción con CatBoost.")
plt.plot(df_rolling, 
         label="Datos originales")
plt.plot(train_predictions_rolling.loc[test.index], 
         label="Predicción")
plt.legend()
plt.show()

# %% [markdown]
# Lista de revisión
# %% [markdown]
"""
- [x] Jupyter Notebook está abierto.
- [ ]  El código no tiene errores
- [ ]  Las celdas con el código han sido colocadas en el orden de ejecución.
- [ ]  Los datos han sido descargados y preparados.
- [ ]  Se ha realizado el paso 2: los datos han sido analizados
- [ ]  Se entrenó el modelo y se seleccionaron los hiperparámetros
- [ ]  Se han evaluado los modelos. Se expuso una conclusión
- [ ]  La *RECM* para el conjunto de prueba no es más de 48"""
# %%
