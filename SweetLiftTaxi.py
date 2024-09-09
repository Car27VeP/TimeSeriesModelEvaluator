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

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import catboost as cb
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
sin embargo no es problema para seguir adelante. El número de viajes totales fueron de 26 496, hubo resgistro hubo cero viajes 
y la cantidad máxima de viajes en una hora y día específico fueron de 119.
"""
# %% [markdown]
### Cambiar formato fecha y hacer remuestreo por una hora.
# %%
df_taxi['datetime'] = pd.to_datetime(df_taxi['datetime'])
df = df_taxi.set_index('datetime')
df = df.resample('1D').sum()
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
# %% [markdown]
## 3. Formación
# %% [markdown]
## Prueba
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
