"""Carga de los datos del caso y construcción de la demanda limpia de Kaido.

Todas las series quedan indexadas por fecha ficticia (A001 -> 2001, ...),
con frecuencia mensual de inicio de mes ('MS').
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"

SEGMENTOS = ["MSUV", "SSUV", "C", "B", "CSUV", "MD", "S"]
SUCESOS = ["incentivo_publico", "adelanto_compras", "caida_tras_adelanto", "subida_iva"]


def leer(fichero):
    # Solo las celdas vacías son datos que faltan (el código de modelo "NA" no es nulo)
    return pd.read_csv(DATOS / fichero, keep_default_na=False, na_values=[""])


def fecha(df):
    anio = 2000 + df["anio"].str[1:].astype(int)
    return pd.to_datetime(dict(year=anio, month=df["mes"], day=1))


def cargar():
    """Devuelve un dict con todos los ficheros, ya con la columna 'fecha'."""
    d = {}
    for nombre in ["ventas_mensual", "stock_mensual", "mercado_mensual",
                   "macro_mensual", "calendario_A013"]:
        df = leer(nombre + ".csv")
        df["fecha"] = fecha(df)
        d[nombre] = df
    d["segmentos"] = leer("segmentos.csv").set_index("segmento")
    d["codigos"] = leer("codigos.csv")
    return d


def macro(d):
    """Variables macro A001-A012 + calendario A013, indexado por fecha."""
    m = d["macro_mensual"].set_index("fecha").asfreq("MS")
    cal = d["calendario_A013"].set_index("fecha")[["dias_laborables", "semana_santa"]]
    return pd.concat([m, cal]).asfreq("MS")


def panel_segmento(d):
    """Una fila por mes y segmento con mercado, ventas Kaido, demanda limpia y stock.

    Demanda limpia = ventas retail - automatriculaciones (no son clientes).
    """
    v = (d["ventas_mensual"]
         .groupby(["fecha", "segmento"])[["unidades", "u_automatriculacion", "u_venta_especial"]]
         .sum()
         .rename(columns={"unidades": "ventas", "u_automatriculacion": "automat",
                          "u_venta_especial": "venta_especial"}))
    renov = d["ventas_mensual"].groupby(["fecha", "segmento"])["renovacion_gama"].max()
    m = (d["mercado_mensual"].groupby(["fecha", "segmento"])["unidades"].sum()
         .rename("mercado"))
    s = (d["stock_mensual"].set_index(["fecha", "segmento"])
         [["stock_total", "s_0_90", "s_91_180", "s_mas_180", "s_sin_cliente", "wholesale"]])

    p = pd.concat([m, v, renov, s], axis=1).sort_index()
    p["demanda"] = p["ventas"] - p["automat"]
    p["cuota"] = p["ventas"] / p["mercado"]
    p["cuota_limpia"] = p["demanda"] / (p["mercado"] - p["automat"])
    return p


if __name__ == "__main__":
    d = cargar()
    p = panel_segmento(d)
    print(p.loc["2012"].groupby("segmento")[["mercado", "ventas", "automat", "demanda"]].sum())
