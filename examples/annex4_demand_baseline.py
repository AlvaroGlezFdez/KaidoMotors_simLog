"""Anexo 4 del enunciado: demanda mensual del SUV medio (MSUV).

Pasos:
  1. Separar la serie del mercado en tendencia, estacionalidad y resto (STL).
  2. Entrenar un SARIMAX hasta A011 (año ficticio 2011) y compararlo con A012.

Ejecutar desde la raíz del repositorio:
    python examples/annex4_demand_baseline.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Solo las celdas vacías son datos que faltan (si no, el código de modelo "NA" se lee como nulo)
leer = lambda f: pd.read_csv('data/' + f,
                             keep_default_na=False, na_values=[''])
mercado = leer('mercado_mensual.csv')
macro = leer('macro_mensual.csv')


# A001 pasa a ser el año ficticio 2001, A002 el 2002, ...
def fecha(df):
    anio = 2000 + df['anio'].str[1:].astype(int)
    return pd.to_datetime(dict(year=anio, month=df['mes'], day=1))


mercado['fecha'] = fecha(mercado)
macro['fecha'] = fecha(macro)

# mercado mensual del SUV medio, con frecuencia de inicio de mes
y = (mercado[mercado['segmento'] == 'MSUV']
     .groupby('fecha')['unidades'].sum().asfreq('MS'))
variables = ['paro', 'confianza_consumidor', 'dias_laborables', 'semana_santa']
X = macro.set_index('fecha')[variables].asfreq('MS')

# 1. separar la serie en tendencia, estacionalidad y resto
partes = STL(np.log(y), period=12).fit()
partes.plot()

# 2. entrenar hasta A011 (año ficticio 2011) y comprobar con A012
ent = y.index.year <= 2011
modelo = sm.tsa.SARIMAX(np.log(y[ent]), exog=X[ent], order=(1, 0, 0),
                        seasonal_order=(0, 1, 1, 12)).fit(
    disp=False,
    maxiter=1000,  # En caso de que salga "ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals"
    # Ocurre cuando hay pocos datos de entrenamiento y el modelo no converge. En ese caso, aumentar el número de iteraciones puede ayudar a que el modelo encuentre una solución.
)
prevision = np.exp(modelo.forecast(12, exog=X[~ent]))
