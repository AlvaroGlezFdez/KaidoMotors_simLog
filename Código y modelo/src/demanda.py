"""Modelo de demanda de Kaido (de arriba abajo):

    demanda Kaido = mercado total x peso del segmento x cuota de Kaido

- Mercado total (7 segmentos): SARIMAX sobre log(mercado) con calendario, planes
  de ayuda, subidas del IVA y dos variables económicas retrasadas 3 meses.
  Los meses de adelanto y de caída tras el adelanto se tratan como datos que
  faltan: el modelo aprende la demanda "normal" y los sucesos se aplican aparte.
- Peso del segmento: peso medio del último año, con la tendencia de los dos
  últimos años (los SUV ganan peso, compactos y berlinas lo pierden).
- Cuota de Kaido: cuota limpia (demanda / mercado) media de los últimos 12 meses.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

CALENDARIO = ["dias_laborables", "semana_santa"]
SUCESOS_MODELO = ["incentivo_publico", "subida_iva"]
EVENTOS_FUERA = ["adelanto_compras", "caida_tras_adelanto"]
ECONOMIA = ["confianza_consumidor", "paro"]
RETRASO = 3
ORDEN = (0, 0, 1)
ORDEN_ESTACIONAL = (0, 1, 1, 12)


def exogenas(mac, economia=ECONOMIA, retraso=RETRASO, sucesos=SUCESOS_MODELO):
    """Variables externas por mes. Las económicas, retrasadas 'retraso' meses."""
    X = mac[CALENDARIO + list(sucesos)].copy()
    for v in economia:
        X[f"{v}_l{retraso}"] = mac[v].shift(retraso)
    return X


def meses_evento(mac):
    """True en los meses de adelanto de compras o de caída tras el adelanto."""
    return mac[EVENTOS_FUERA].sum(axis=1) > 0


def ajustar_mercado(y, X, fin_entreno, eventos=None, orden=ORDEN, orden_est=ORDEN_ESTACIONAL):
    """SARIMAX sobre log(y) con datos hasta 'fin_entreno'.

    'eventos': serie booleana; esos meses se dejan como dato que falta (NaN).
    """
    ly = np.log(y.loc[:fin_entreno]).copy()
    if eventos is not None:
        ly[eventos.reindex(ly.index, fill_value=False)] = np.nan
    Xe = X.loc[ly.index]
    Xe = Xe.loc[:, Xe.std() > 0]          # fuera variables que nunca se mueven
    ok = Xe.notna().all(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sm.tsa.SARIMAX(ly[ok], exog=Xe[ok], order=orden,
                              seasonal_order=orden_est).fit(disp=False, maxiter=1000)


def prever_mercado(mod, X, fechas):
    """Previsión en unidades para 'fechas' (meses consecutivos tras el entrenamiento)."""
    return np.exp(mod.forecast(len(fechas), exog=X.loc[fechas, mod.model.exog_names]))


def peso_segmento(mercado_seg, fin_entreno):
    """Peso de cada segmento en el mercado total para el año siguiente.

    Peso medio de los últimos 12 meses x tendencia anual media de los dos últimos años.
    """
    peso = mercado_seg.div(mercado_seg.sum(axis=1), axis=0).loc[:fin_entreno]
    ult = peso.iloc[-12:].mean()
    hace2 = peso.iloc[-36:-24].mean()
    w = ult * (ult / hace2) ** 0.5
    return w / w.sum()                    # los pesos tienen que sumar 100 %


def cuota_media(cuota, fin_entreno, meses=12, excluir=None):
    """Cuota limpia media de los últimos 'meses' meses (sin los meses de 'excluir')."""
    c = cuota.loc[:fin_entreno]
    if excluir is not None:
        c = c.drop(excluir, errors="ignore")
    return c.dropna().iloc[-meses:].mean()


def ingenuo(serie, fechas):
    """Modelo ingenuo estacional: el mismo mes del año anterior."""
    return serie.shift(12).loc[fechas]


def error_pct(real, prev):
    """Error del total en %: positivo = el modelo se pasa."""
    return 100 * (prev.sum() / real.sum() - 1)


def mape(real, prev):
    """Error mensual medio en valor absoluto (%)."""
    return 100 * (np.abs(prev / real - 1)).mean()


def sigma_log(real, prev):
    """Error típico mensual en escala logarítmica (0,10 ≈ 10 %)."""
    e = np.log(real / prev)
    return float(np.sqrt((e ** 2).mean()))


# --- Cuota: métodos alternativos y previsión que rueda ---------------------------

def cuota_exponencial(cuota, fin, alfa=0.3):
    """Media con más peso a los meses recientes (suavizado exponencial)."""
    c = cuota.loc[:fin].dropna()
    return c.ewm(alpha=alfa, adjust=False).mean().iloc[-1]


def cuota_desde_renovacion(cuota, renov, fin, maximo=12, minimo=3):
    """Media desde la última renovación de gama (entre 'minimo' y 'maximo' meses)."""
    c = cuota.loc[:fin].dropna()
    r = renov.reindex(c.index).fillna(0)
    ult = r[r > 0].index.max() if (r > 0).any() else None
    n = maximo if ult is None else len(c.loc[ult:])
    n = int(min(max(n, minimo), maximo))
    return c.iloc[-n:].mean()


def mercado_rodante(mod, y, X, eventos, objetivos, h=3):
    """Previsión a 'h' meses vista: para cada mes objetivo se usa la información hasta h meses antes.

    Los parámetros del modelo se mantienen fijos (los del entrenamiento); solo se actualizan los datos.
    """
    ly = np.log(y).copy()
    ly[eventos.reindex(ly.index, fill_value=False)] = np.nan
    cols = mod.model.exog_names
    out = {}
    for f in objetivos:
        origen = f - pd.DateOffset(months=h)
        hist = ly.loc[:origen]
        Xh = X.loc[hist.index, cols]
        ok = Xh.notna().all(axis=1)
        res = mod.apply(hist[ok], exog=Xh[ok])
        fechas = pd.date_range(origen + pd.DateOffset(months=1), periods=h, freq="MS")
        out[f] = float(np.exp(res.forecast(h, exog=X.loc[fechas, cols]).iloc[-1]))
    return pd.Series(out)
