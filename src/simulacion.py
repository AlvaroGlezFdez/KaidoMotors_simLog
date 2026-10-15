"""Previsión de A013 por escenario y simulación del año con el plan que rueda.

Para cada escenario y cada segmento se simulan R años (Monte Carlo):
- demanda real = demanda esperada del escenario x e^(sesgo + ruido)
- cada mes se revisa la previsión con lo observado y se decide la llegada de t+3:
      L[t+3] = max(0, D^[t+3] + SS[t+3] - S^[t+2])
- se vende min(demanda, stock disponible); lo que falta es transfer o venta perdida
- el stock se sigue por edad (FIFO) para el VM y los coches de más de 90 días.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from datos import SEGMENTOS
from demanda import (exogenas, meses_evento, ajustar_mercado, prever_mercado,
                     peso_segmento, cuota_media)

MESES = pd.date_range("2013-01-01", periods=12, freq="MS")
NOMBRES_MES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
COSTE_MES = 2.0 * 30          # € por coche parado al mes para la marca (hasta 180 días)
EDADES = 24                   # meses de antigüedad que se siguen
PESO_COMUN = 0.5              # parte de la varianza del sesgo común a todos los segmentos
SEGMENTOS_NUEVOS = ["B", "SSUV", "CSUV", "MSUV"]   # segmentos con códigos nuevos en A013

ESCENARIOS = {
    "Pesimista": dict(d_paro=+2.0, meses_caida=6, plan_ayuda_desde=None, cuota_nuevos=0.85,
                      meses_cuota=3, prima_co2=True, arancel=+0.03, rotura=("MSUV", 5)),
    "Central": dict(d_paro=0.0, meses_caida=3, plan_ayuda_desde=None, cuota_nuevos=1.0,
                    meses_cuota=0, prima_co2=False, arancel=0.0, rotura=None),
    "Optimista": dict(d_paro=-1.0, meses_caida=1, plan_ayuda_desde=7, cuota_nuevos=1.10,
                      meses_cuota=12, prima_co2=False, arancel=-0.02, rotura=None),
}


# --- Economía de cada coche --------------------------------------------------------

def economia_coche(seg):
    """Euros por coche y segmento (de segmentos.csv)."""
    e = pd.DataFrame(index=SEGMENTOS)
    e["bb"] = seg["beneficio_bruto_marca_eur"]
    e["vm0"] = seg["vm_real_0_90_eur"]
    e["vm1"] = seg["vm_real_91_180_eur"]
    e["vm2"] = seg["vm_real_mas_180_eur"].fillna(seg["vm_real_91_180_eur"])
    e["prima_co2"] = seg["prima_co2_eur"]
    e["coste_fabricar"] = seg["pff_medio"] - seg["margen_bruto_conc_eur"] - seg["beneficio_bruto_marca_eur"]
    e["cvp_transfer"] = 0.8 * e["bb"] - e["vm0"]
    return e


def z_por_segmento(econ):
    """Nivel de servicio del 'vendedor de periódicos': z = Φ⁻¹(Cu / (Cu + Co)).

    Cu: coste de quedarse corto un coche (transfer si compensa; si no, el CVP perdido).
    Co: coste de que sobre un coche unos 3 meses (stock + más VM por envejecer).
    """
    cu = np.where(econ["cvp_transfer"] > 0, 0.2 * econ["bb"], np.maximum(econ["bb"] - econ["vm0"], 0))
    co = 3 * COSTE_MES + np.maximum(econ["vm1"] - econ["vm0"], 0)
    ratio = cu / (cu + co)
    z = np.where(cu > 0, np.maximum(norm.ppf(ratio), 0), 0.0)
    return pd.DataFrame({"Cu": cu, "Co": co, "ratio_critico": ratio, "z": z}, index=econ.index)


# --- Previsión de A013 ------------------------------------------------------------

def exogenas_escenario(mac, esc):
    """Variables externas de A013 bajo un escenario."""
    m = mac.copy()
    a013 = m.index >= "2013-01-01"
    ult = m.loc["2012-12-01"]
    m.loc[a013, "paro"] = ult["paro"] + esc["d_paro"]
    m.loc[a013, "confianza_consumidor"] = ult["confianza_consumidor"]
    for v in ["incentivo_publico", "subida_iva", "adelanto_compras", "caida_tras_adelanto"]:
        m.loc[a013, v] = 0
    if esc["plan_ayuda_desde"]:
        m.loc[MESES[esc["plan_ayuda_desde"] - 1:], "incentivo_publico"] = 1
    return exogenas(m)


def prevision_a013(d, p, mac):
    """Demanda esperada de A013 (12 x 7) para cada escenario, y piezas intermedias."""
    mercado = p["mercado"].unstack("segmento").asfreq("MS")[SEGMENTOS]
    demanda = p["demanda"].unstack("segmento").asfreq("MS")[SEGMENTOS]
    total = mercado.sum(axis=1)
    cuota = demanda / mercado
    eventos = meses_evento(mac)
    X0 = exogenas(mac)
    mod = ajustar_mercado(total, X0, "2012-12-01", eventos=eventos)

    # Tamaño del adelanto de nov-dic A012: real menos lo que el modelo esperaba sin adelanto
    esperado = np.exp(mod.predict(start="2012-11-01", end="2012-12-01"))
    adelantados = float(total["2012-11":"2012-12"].sum() - esperado.sum())

    peso = peso_segmento(mercado, "2012-12-01")
    censurados = {"SSUV", "S", "MSUV"}           # diciembre A012 sin coches
    cu = {}
    for s in SEGMENTOS:
        if s == "CSUV":
            cu[s] = cuota[s].loc["2012-12-01"]   # lanzamiento: última cuota observada
        else:
            excl = [pd.Timestamp("2012-12-01")] if s in censurados else None
            cu[s] = cuota_media(cuota[s], "2012-12-01", 12, excluir=excl)
    cu = pd.Series(cu)

    prev, mercados = {}, {}
    central_total = None
    for nombre in ["Central", "Pesimista", "Optimista"]:
        esc = ESCENARIOS[nombre]
        X = exogenas_escenario(mac, esc)
        mt = prever_mercado(mod, X, MESES)
        if central_total is None:
            central_total = mt
            hondura = adelantados / central_total.iloc[:3].sum()   # caída mensual si se devuelve en 3 meses
        caida = np.ones(12)
        caida[:esc["meses_caida"]] = 1 - hondura
        mt = mt * caida
        mercados[nombre] = mt
        dem = pd.DataFrame({s: mt * peso[s] * cu[s] for s in SEGMENTOS})
        if esc["cuota_nuevos"] != 1.0:
            dem.iloc[:esc["meses_cuota"], [SEGMENTOS.index(s) for s in SEGMENTOS_NUEVOS]] *= esc["cuota_nuevos"]
        prev[nombre] = dem
    info = dict(modelo=mod, adelantados=adelantados, hondura=hondura, peso=peso, cuota=cu,
                mercado=pd.DataFrame(mercados))
    return prev, info


# --- Situación de partida ----------------------------------------------------------

def situacion_inicial(d):
    """Stock a 31-dic-A012 por edad y llegadas ya en camino para enero y febrero."""
    st = d["stock_mensual"]
    dic = st[(st.anio == "A012") & (st.mes == 12)].set_index("segmento").loc[SEGMENTOS]
    stock = np.zeros((len(SEGMENTOS), EDADES))
    stock[:, 1] = dic["s_0_90"]          # ≈ 1 mes
    stock[:, 4] = dic["s_91_180"]        # ≈ 4 meses
    stock[:, 7] = dic["s_mas_180"]       # ≈ 7 meses
    ult = st[(st.anio == "A012") & (st.mes >= 10)].groupby("segmento")["wholesale"].mean()
    en_camino = ult.loc[SEGMENTOS].round().to_numpy()
    return stock, en_camino


# --- Simulación -----------------------------------------------------------------------

def simular_segmento(d_plan, d_esc, stock0, en_camino, sigma_ruido, sigma_sesgo, z, econ_s,
                     esc, seg, R=1000, alfa=0.3, rng=None, con_ruido=True, choque_comun=None):
    """Simula R años de un segmento. Devuelve dict de arrays (R,) y llegadas (R, 12).

    choque_comun: normal estándar (R,) compartida por todos los segmentos del mismo año simulado.
    La mitad de la varianza del sesgo es común (todos fallan a la vez) y la otra mitad, propia.
    """
    rng = rng or np.random.default_rng(0)
    T = 12
    if con_ruido:
        comun = choque_comun if choque_comun is not None else rng.normal(size=R)
        propio = rng.normal(size=R)
        b = (sigma_sesgo * (np.sqrt(PESO_COMUN) * comun + np.sqrt(1 - PESO_COMUN) * propio)
             - sigma_sesgo ** 2 / 2)[:, None]
        eta = rng.normal(-sigma_ruido ** 2 / 2, sigma_ruido, (R, T))
        D = d_esc[None, :] * np.exp(b + eta)
    else:
        D = np.repeat(d_esc[None, :], R, axis=0)
    sigma_tot = np.sqrt(sigma_ruido ** 2 + sigma_sesgo ** 2)
    ext = np.concatenate([d_plan, d_plan[-3:]])          # previsión más allá de diciembre
    SS = z * sigma_tot * ext * np.sqrt(3)

    # economía por coche en este escenario
    bb = econ_s["bb"] - econ_s["coste_fabricar"] * esc["arancel"]
    co2 = econ_s["prima_co2"] if esc["prima_co2"] else 0.0
    vm = np.array([econ_s["vm0"], econ_s["vm1"], econ_s["vm2"]])
    hay_transfer = (0.8 * bb - econ_s["vm0"] - co2) > 0

    stock = np.repeat(stock0[None, :], R, axis=0).astype(float)
    L = np.zeros((R, T + 3))
    L[:, 0] = L[:, 1] = en_camino
    logk = np.zeros(R)                                  # corrección de la previsión (log)

    def pedir(k):
        """Al cerrar el mes k (−1 = dic A012) se decide la llegada de k+3."""
        f = np.exp(logk)[:, None]
        dhat = ext[None, :] * f
        s_k = stock.sum(axis=1)
        proy = s_k + L[:, k + 1] + L[:, k + 2] - dhat[:, k + 1] - dhat[:, k + 2]
        L[:, k + 3] = np.maximum(0, dhat[:, k + 3] + SS[k + 3] - proy)

    pedir(-1)
    ventas = np.zeros((R, T)); transf = np.zeros((R, T)); perdidas = np.zeros((R, T))
    vm_total = np.zeros(R); coste_stock = np.zeros(R); stock_fin = np.zeros((R, T))
    vendidos_edad = np.zeros((R, 3))
    for t in range(T):
        if esc["rotura"] and esc["rotura"][0] == seg and t == esc["rotura"][1]:
            L[:, t] = 0                                 # la fábrica pierde un mes de producción
        stock[:, 0] += L[:, t]
        disp = stock.sum(axis=1)
        v = np.minimum(D[:, t], disp)
        falta = D[:, t] - v
        tr = falta if hay_transfer else np.zeros(R)
        ventas[:, t], transf[:, t], perdidas[:, t] = v, tr, falta - tr
        # FIFO: se vende primero el coche más viejo
        quedan = v.copy()
        for a in range(EDADES - 1, -1, -1):
            x = np.minimum(stock[:, a], quedan)
            stock[:, a] -= x
            quedan -= x
            vendidos_edad[:, 0 if a < 3 else (1 if a < 6 else 2)] += x
        stock_fin[:, t] = stock.sum(axis=1)
        coste_stock += COSTE_MES * stock[:, :6].sum(axis=1)
        # revisar la previsión con la demanda observada: los concesionarios registran también
        # los pedidos que no pudieron servir (si no, la previsión se hunde sola: demanda censurada)
        obs = np.maximum(D[:, t], 1)
        logk = (1 - alfa) * logk + alfa * np.log(obs / d_plan[t])
        # envejecer el stock un mes
        stock[:, 1:] = stock[:, :-1].copy(); stock[:, 0] = 0
        if t <= 8:
            pedir(t)

    vm_total = vendidos_edad @ vm + transf.sum(axis=1) * econ_s["vm0"]
    cvp = (ventas.sum(axis=1) * (bb - co2) + transf.sum(axis=1) * (0.8 * bb - co2)) - vm_total
    return dict(
        vendidos=ventas.sum(axis=1) + transf.sum(axis=1),
        llegadas=L[:, :T].sum(axis=1),
        stock_medio=stock_fin.mean(axis=1),
        meses_venta=stock_fin.mean(axis=1) / np.maximum((ventas + transf).mean(axis=1), 1),
        mas_90=stock[:, 4:].sum(axis=1),       # tras envejecer: 4+ meses en dic = >90 días a 31-dic
        coste_stock=coste_stock,
        perdidas=perdidas.sum(axis=1),
        perdidas_eur=perdidas.sum(axis=1) * np.maximum(bb - co2 - econ_s["vm0"], 0),
        transfers=transf.sum(axis=1),
        transfers_eur=transf.sum(axis=1) * 0.2 * bb,
        vm=vm_total,
        cvp=cvp,
        margen=cvp - coste_stock,
        L=L[:, :T],
        D=D,
    )


def simular(prev, econ, zs, sigmas, sigma_sesgo, stock0, en_camino, R=1000, semilla=13):
    """Simula los tres escenarios. Siempre se planifica con la previsión central."""
    rng = np.random.default_rng(semilla)
    plan = prev["Central"]
    res = {}
    for nombre, d_esc in prev.items():
        res[nombre] = {}
        comun = rng.normal(size=R)                      # mismo choque para los 7 segmentos
        for i, s in enumerate(SEGMENTOS):
            res[nombre][s] = simular_segmento(
                plan[s].to_numpy(), d_esc[s].to_numpy(), stock0[i], en_camino[i],
                sigmas[s], sigma_sesgo, zs.loc[s, "z"], econ.loc[s], ESCENARIOS[nombre], s, R=R, rng=rng,
                choque_comun=comun)
    return res


def plan_central(prev, econ, zs, sigmas, sigma_sesgo, stock0, en_camino):
    """Plan de llegadas si el año sale exactamente como la previsión central."""
    filas = {}
    for i, s in enumerate(SEGMENTOS):
        r = simular_segmento(prev["Central"][s].to_numpy(), prev["Central"][s].to_numpy(), stock0[i],
                             en_camino[i], sigmas[s], sigma_sesgo, zs.loc[s, "z"], econ.loc[s],
                             ESCENARIOS["Central"], s, R=1, con_ruido=False)
        filas[s] = r["L"][0]
    return pd.DataFrame(filas, index=NOMBRES_MES).round().astype(int)


METRICAS = [
    ("vendidos", "Coches vendidos (unidades)"),
    ("llegadas", "Coches fabricados / llegadas (unidades)"),
    ("stock_medio", "Stock medio en la red (coches)"),
    ("meses_venta", "Stock medio en la red (meses de venta)"),
    ("mas_90", "Coches con más de 90 días a fin de año"),
    ("coste_stock", "Coste del stock para la marca (€)"),
    ("perdidas", "Ventas perdidas (coches)"),
    ("perdidas_eur", "Ventas perdidas (€ de CVP)"),
    ("transfers", "Transfers (coches)"),
    ("transfers_eur", "Transfers (€ de coste)"),
    ("vm", "Gasto comercial, VM (€)"),
    ("cvp", "Margen de la marca, CVP (€)"),
]


def tabla5(res, segmento=None, estadistico="media"):
    """Tabla 5 del enunciado: total (o un segmento), media de los R años simulados."""
    f = {"media": np.mean, "p10": lambda x: np.percentile(x, 10), "p90": lambda x: np.percentile(x, 90)}[estadistico]
    out = {}
    for esc in ["Pesimista", "Central", "Optimista"]:
        col = {}
        segs = [segmento] if segmento else SEGMENTOS
        for k, nombre in METRICAS:
            if k == "meses_venta":
                st = sum(res[esc][s]["stock_medio"] for s in segs)
                vt = sum(res[esc][s]["vendidos"] for s in segs) / 12
                col[nombre] = f(st / np.maximum(vt, 1))
            else:
                col[nombre] = f(sum(res[esc][s][k] for s in segs))
        out[esc] = col
    return pd.DataFrame(out)
