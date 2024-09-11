# %% [markdown]
"""
# Descripción del proyecto
La compañía Sweet Lift Taxi ha recopilado datos históricos sobre 
pedidos de taxis en los aeropuertos. Para atraer a más conductores durante
las horas pico, necesitamos predecir la cantidad de pedidos de taxis para la próxima hora. 
Construye un modelo para dicha predicción.

La métrica RECM en el conjunto de prueba no debe ser superior a 48.

## Instrucciones del proyecto.

1. Descarga los datos y haz el remuestreo por una hora.
2. Analiza los datos
3. Entrena diferentes modelos con diferentes hiperparámetros. 
        La muestra de prueba debe ser el 10% del conjunto de datos inicial.
4. Prueba los datos usando la muestra de prueba y proporciona una conclusión.

## Descripción de los datos

Los datos se almacenan en el archivo `taxi.csv`. 	
El número de pedidos está en la columna `num_orders`.
"""
# %% [markdown]
"""
# 1. Preparacion
"""
# %%
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# %% [markdown]
"""
# Carga de los datos.
"""
# %%
df_taxi = pd.read_csv(r'../../../datasets/taxi.csv')
# %% [markdown]
"""
# 2. Análisis
"""
# %% [markdown]
"""
## Observación general de los datos.
"""
# %%
df_taxi.head()
# %%
df_taxi.info()
# %% [markdown]
"""
Obervamos que los datos se cargaron correctamente. 
El DataFrame 26,406 filas por 2 columnas, del cual no hay ningún registro nulo.
el tipo de valores que tiene nuestro dataframe es object e int64. Dado que tenemos una columa llamada datetime,
cambiar esta columna al formato correspondiente. También observamos el número de viajes registrados es
cada 10 minutos.
"""
# %% [markdown]
"""
## Obervación estadística de los datos antes del remuestreo.
"""
# %%
df_taxi.describe()
# %%
df_taxi.plot(kind='box', title='Distribución de los viajes.')
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
"""
## Cambiar formato fecha y hacer remuestreo por una hora.
"""
# %%
df_taxi['datetime'] = pd.to_datetime(df_taxi['datetime'])
df_taxi = df_taxi.set_index('datetime')
df_taxi = df_taxi.resample('1D').sum()
df = df_taxi.copy()
df['num_orders'].rolling(5).mean().plot(title="Viajes de marzo a agosto del 2018.")
plt.show()
# %% [markdown]
"""
## Obervación estadística de los datos antes del remuestreo.
"""
# %%
df_taxi.describe()
# %%
df_taxi.plot(kind='box', title='Distribución de los viajes.')
plt.show()
# %% [markdown]
"""
En la gráfica podemos observar la cantidad de viajes regitrados
de los días de marzo a agosto. Además suavizamos con promedio móvil la gráfica.
"""
# %%
# Tendencia, estacionalidad y residuales.
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
"""
## Hacer los datos mas estacionarios.
"""
# %%
data = df[['num_orders']].copy()
data -= data.shift()

data['mean'] = data['num_orders'].rolling(15).mean()
data['std'] = data['num_orders'].rolling(15).std()

data.plot(title='Datos más estacionarios.')
plt.show()
# %% [markdown]
"""
### Escalar datos.
"""
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
# %% [markdown]
"""
# Funciones.
"""
# %%
# Gráfica promedio movil
def grafica_promedio_movil(data, predictions, target_train, target_test, title=''):
    pred_test = pd.Series(predictions, target_test.index)
    train_with_predictions = pd.concat([target_train, pred_test])

    train_predictions_rolling = train_with_predictions.rolling(15).mean()
    df_rolling = data['num_orders'].rolling(15).mean()

    plt.title(title)
    plt.plot(df_rolling, label="Datos originales")
    plt.plot(train_predictions_rolling.loc[test.index],
             label="Predicción")
    plt.legend()
    plt.show()
    
# Grafica RMSE
def grafica_rmse(rmse_list, title=''):
    plt.plot(rmse_list)
    plt.plot(np.argmin(rmse_list), np.min(rmse_list),
             marker='*', markersize=15,
             label=f"Mejor RMSE: {np.min(rmse_list):.2f}")
    plt.title(title)
    plt.legend()
    plt.show()
    
# Creación de características para los modelos.
def make_features(data, max_lag, rolling_mean_size):
    new_data = data.copy()
    new_data['year'] = new_data.index.year
    new_data['month'] = new_data.index.month
    new_data['day'] = new_data.index.day
    new_data['dayofweek'] = new_data.index.dayofweek

    for lag in range(1, max_lag + 1):
        new_data['lag_{}'.format(lag)] = new_data['num_orders'].shift(lag)

    new_data['rolling_mean'] = (
        new_data['num_orders'].shift().rolling(rolling_mean_size).mean()
    )
    
    return new_data

# Dividir el conjunto de datos en formato de entrenamiento (90%) y prueba (10%) sin aletoriedad.
def splitting_data(data):
    
    train, test = train_test_split(data,test_size=0.1,shuffle=False,random_state=1234)
    train = train.dropna()
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    target_test = test[['num_orders']]
    
    return features_train,target_train,features_test,target_test

# %% [markdown]
"""
## 3. Formación
"""
# %% [markdown]
"""
### Prueba
"""
# %% [markdown]
"""
### Pruebas de cordura.
"""
# %%
train, test = train_test_split(df, test_size=0.1, shuffle=False)
# %% [markdown]
"""
### Método 1.
"""
# %%
print(f"Viajes medios al día: {np.median(df)}")
pred_median = np.ones(test.shape) * np.median(train)
print('RMSE:', rmse(test, pred_median))
# %% [markdown]
"""
### Método 2.
"""
# %%
print(f"Viajes medios al día: {np.median(df)}")
pred_previous = test.shift()
pred_previous.iloc[0] = train.iloc[-1]
print('RMSE:', rmse(test, pred_previous))
# %% [markdown]
"""
Nuestras prueba de cordura 1 tuvo un error demasiado grande, por otro lado, al segundo método, que aplicamos shifting,
le fue mejor pero no llegamos al objetivo RMSE que debe es 49. 
En los siguientes modelos haremos que este valor sea menor creando caracterísitcas a partir de la serie temporal.
"""
# %% [markdown]
print("Antes de agregar características:\n")
print(df)
#make_features(df, 10, 80)
featured_df = make_features(df, 10, 14)
print("Depués de agregar características\n")
print(featured_df)
# data_features = data.drop('num_orders',axis=1)
# data_target = data['num_orders']
train, test = train_test_split(featured_df,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
train = train.dropna()


features_train = train.drop(['num_orders'], axis=1)
target_train = train[['num_orders']]
features_test = test.drop(['num_orders'], axis=1)
target_test = test[['num_orders']]
# %% [markdown]
"""
# Regresión Lineal.
"""
# %%
rmse_list = []
for x in range(40):
    new_data = make_features(df,2*(x+1),4*(x+1))
    features_train, target_train, features_test, target_test = splitting_data(new_data)

    model = LinearRegression()
    model.fit(features_train, target_train)
    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)}:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
grafica_rmse(rmse_list, title="RMSE Regresión Lineal con 30 iteraciones")
# %%
rmse_list = []
for x in range(40):
    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    train = train.dropna()
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    target_test = test[['num_orders']]

    model = LinearRegression()
    model.fit(features_train, target_train)
    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)}:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
grafica_rmse(rmse_list, title="RMSE Regresión Lineal con 30 iteraciones")
# %% [markdown]
# Árboles de Decisión
# %%
rmse_list = []
pred_test_list = []

for x in range(30):
    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    train = train.dropna()
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    target_test = test[['num_orders']]
    
    
    model = DecisionTreeRegressor(max_depth=x+1,
                                  random_state=1234)
    model.fit(features_train, target_train)

    pred_train = model.predict(features_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)}: {rmse(target_test, pred_test):.2f}')

    rmse_list.append(rmse(target_test, pred_test))
    pred_test_list.append(np.array(pred_test))
rmse_list = np.array(rmse_list)
pred_test_list = np.array(pred_test_list)
grafica_rmse(rmse_list, title="RMSE Árboles con 30 iteraciones")
# %%
# Random Forest
# %%
rmse_list = []
pred_test_list = []
for x in range(40):

    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    train = train.dropna()
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    target_test = test[['num_orders']]
    
    model = RandomForestRegressor(random_state=12345,
                                  criterion='squared_error',
                                  max_depth=x+1)
    
    model.fit(features_train, target_train.values.ravel())

    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y max_depth {x+1} en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')

    rmse_list.append(rmse(target_test, pred_test))
    pred_test_list.append(np.array(pred_test))

rmse_list = np.array(rmse_list)
pred_test_list = np.array(pred_test_list)

grafica_rmse(rmse_list, title="")
# grafica_promedio_movil(data, pred_test_list[np.argmin(
#     rmse_list)-1], target_train, target_test, title="Promedio movil predicción con Random Forest")
# %% [markdown]
# CatBoost
# %%
rmse_list = []
for x in range(30):
    
    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    train = train.dropna()
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    target_test = test[['num_orders']]
    
    model = CatBoostRegressor()
    # Con optimización automatizada de hiperparámetros. Facilita la búsqueda del mejor hiperparametros para un modelo.
    parameters = {'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
                }

    grid = GridSearchCV(estimator=model, param_grid=parameters, cv=2, n_jobs=-1)
    grid.fit(features_train, target_train)

    print(f"Los mejores parametros para CatBoost: {grid.best_params_}")
    model = CatBoostRegressor(**grid.best_params_)
    model.fit(features_train, target_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y {grid.best_params_} en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
rmse_list = np.array(rmse_list)
# %%
grafica_rmse(rmse_list)
# %% [markdown]
# LGBMRegressor
# %%
rmse_list = []
for x in range(10):
    
    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    train = train.dropna()
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    target_test = test[['num_orders']]
    
    model = LGBMRegressor()

    param_space = {
        'max_depth': (3, 12),
        'num_leaves': (20, 150),
        'learning_rate': (1e-4, 1e-1, 'log-uniform'),
        'min_data_in_leaf': (50, 300)
    }

    opt = BayesSearchCV(model, param_space, n_iter=50, cv=5, n_jobs=-1)
    opt.fit(features_train, target_train)

    print("Los mejores parametros para LightGBM: ", opt.best_params_)
    
    model = LGBMRegressor(**dict(opt.best_params_))
    model.fit(features_train, target_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y {opt.best_params_} en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
rmse_list = np.array(rmse_list)
grafica_rmse(rmse_list)
# %% [markdown]
### XGBoost
# %%
rmse_list = []
for x in range(30):

    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    train = train.dropna()
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)

    model =  XGBRegressor(alpha=0.1,
                          max_depth=5, 
                          eta=0.1, 
                          subsample=0.7, 
                          colsample_bytree=0.7, 
                          n_estimators = 100)
    model.fit(features_train, target_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y con los hiperparametros seleccionados en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
rmse_list = np.array(rmse_list)
grafica_rmse(rmse_list)
# %% [markdown]
### AdaBoostRegressor 
# %%
rmse_list = []
for x in range(30):

    new_data = make_features(df,2*(x+1),4*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
    
    train = train.dropna()  
    
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=(x+1)),
                            n_estimators=1700, 
                            learning_rate=0.6, 
                            loss='exponential', random_state=42)
    model.fit(features_train, target_train.values.ravel())
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y con los hiperparametros seleccionados en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
rmse_list = np.array(rmse_list)
grafica_rmse(rmse_list)
# %% [markdown]
### SVR
# %%
rmse_list = []
for x in range(80):

    new_data = make_features(df,1*(x+1),2*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    train = train.dropna() 
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1) 
    
    model = SVR(C=1.0, epsilon=0.2)
    model.fit(features_train, target_train)
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y con los hiperparametros seleccionados en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
rmse_list = np.array(rmse_list)
grafica_rmse(rmse_list)
# %% [markdown]
### TODO MLP
from sklearn.neural_network import MLPRegressor
# %%
rmse_list = []
for x in range(30):
    
    new_data = make_features(df,1*(x+1),2*(x+1))
    train, test = train_test_split(new_data,
                              test_size=0.1,
                               shuffle=False,
                               random_state=1234)
    
    train = train.dropna() 
    
    features_train = train.drop(['num_orders'], axis=1)
    target_train = train[['num_orders']]
    features_test = test.drop(['num_orders'], axis=1)
         
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 100)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001],
        'learning_rate_init': [0.001, 0.01]
    }

    grid_search = GridSearchCV(MLPRegressor(random_state=1, max_iter=500), param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(features_train, target_train)

    print("Mejores parámetros encontrados:", grid_search.best_params_)    
    
    model = MLPRegressor(**grid_search.best_params_)
    model.fit(features_train, target_train.values.ravel())
    pred_test = model.predict(features_test)
    
    print(f'RMSE para el conjunto de prueba con max_lag {2*(x+1)} y max_rolling {4*(x+1)} en los datos y con los hiperparametros seleccionados en el modelo:' 
          f' {rmse(target_test, pred_test):.2f}')
    
    rmse_list.append(rmse(target_test, pred_test))
    
rmse_list = np.array(rmse_list)
grafica_rmse(rmse_list)
    
# pred_test = model.predict(features_test)
# print(F"RMSE MLP: {rmse(target_test,pred_test)}")
# %%
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, features_train, target_train, cv=5, scoring='neg_root_mean_squared_error')
# print("RMSE promedio:", -scores.mean())
# %% [markdown]
### Lasso
# %%
from sklearn import linear_model
model = linear_model.Lasso(alpha=0.1)
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(F"RMSE Lasso: {rmse(target_test,pred_test)}")
# %% [markdown]
from sklearn.linear_model import Ridge
model = Ridge()
param = {'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100],
         'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
grid_search = GridSearchCV(model, param, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(features_train, target_train)
print("Best value for lambda : ",model.get_params())
best_param = model.get_params()
model = Ridge(**best_param)
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(F"RMSE Ridge: {rmse(target_test,pred_test)}")
# %% [markdown]
### ARDRegression
# %%
from sklearn import linear_model
model = linear_model.ARDRegression(max_iter=800)
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(F"RMSE Ridge: {rmse(target_test,pred_test)}")
# %% [markdown]
### Poisson
# %%
from sklearn import linear_model
model = linear_model.PoissonRegressor()
model.fit(features_train, target_train.values.ravel())
pred_test = model.predict(features_test)
print(F"RMSE Poisson: {rmse(target_test,pred_test)}")
# %% [markdown]
### Perceptron
# %%
from sklearn.linear_model import Perceptron
model = Perceptron(tol=1e-3, random_state=0, n_jobs=-1)
model.fit(features_train, target_train.values.ravel())
pred_test = model.predict(features_test)
print(F"RMSE Poisson: {rmse(target_test,pred_test)}")
# %% [markdown]
### ElasticNetCV
# %%
from sklearn.linear_model import ElasticNetCV
model =ElasticNetCV()
model.fit(features_train, target_train.values.ravel())
pred_test = model.predict(features_test)
print(F"RMSE ElasticNetCV: {rmse(target_test,pred_test)}")
# %% [markdown]
### TODO: HuberRegressor
# %%
from sklearn.linear_model import HuberRegressor
model = HuberRegressor(fit_intercept=False)
model.fit(features_train, target_train.values.ravel())
pred_test = model.predict(features_test)
print(F"RMSE HuberRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### RANSACRegressor
# %%
from sklearn.linear_model import RANSACRegressor
model = RANSACRegressor()
model.fit(features_train, target_train.values.ravel())
pred_test = model.predict(features_test)
print(f"RMSE RANSACRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### KNeighborsRegressor
# %%
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=30,weights='distance')
model.fit(features_train, target_train.values.ravel())
pred_test = model.predict(features_test)
print(F"RMSE KNeighborsRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### ExtraTreeRegressor
# %%
from sklearn.tree import ExtraTreeRegressor
model = ExtraTreeRegressor(max_depth=1120,min_weight_fraction_leaf=0.2)
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE ExtraTreeRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### HistGradientBoostingRegressor
# %%
from sklearn.ensemble import HistGradientBoostingRegressor
model = HistGradientBoostingRegressor()
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE HistGradientBoostingRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### StackingRegressor
# %%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
estimators = [
    ('lr', RidgeCV()),
    ('svr', LinearSVR(random_state=42))
]
reg = StackingRegressor(
    estimators=estimators,
    final_estimator=RandomForestRegressor(n_estimators=50,
                                          random_state=42)
)
reg.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE StackingRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### BayesianRidge()

# %%
from sklearn import linear_model
model = linear_model.BayesianRidge()
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE BayesianRidge: {rmse(target_test,pred_test)}")

# %% [markdown]
### TheilSenRegressor
# %% 
from sklearn.linear_model import TheilSenRegressor
model = TheilSenRegressor()
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE TheilSenRegressor: {rmse(target_test,pred_test)}")
# %% [markdown]
### TODO MultiTaskLassoCV
# %%
from sklearn.linear_model import MultiTaskLassoCV
model = MultiTaskLassoCV(cv=2)
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE MultiTaskLassoCV: {rmse(target_test,pred_test)}")
# %% [markdown]
### SGDClassifier
# %%
from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.fit(features_train, target_train)
pred_test = model.predict(features_test)
print(f"RMSE MultiTaskLassoCV: {rmse(target_test,pred_test)}")
# %% [markdown]
# Lista de revisión
# %% [markdown]
"""
- [x] Jupyter Notebook está abierto.
- [x]  El código no tiene errores
- [x]  Las celdas con el código han sido colocadas en el orden de ejecución.
- [x]  Los datos han sido descargados y preparados.
- [x]  Se ha realizado el paso 2: los datos han sido analizados
- [ ]  Se entrenó el modelo y se seleccionaron los hiperparámetros
- [ ]  Se han evaluado los modelos. Se expuso una conclusión
- [ ]  La *RECM* para el conjunto de prueba no es más de 48"""
# %%
