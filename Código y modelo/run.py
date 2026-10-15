"""Pipeline completo del caso Kaido Motors: de los datos a la Tabla 5.

Ejecutar desde la raíz del repositorio:
    python run.py

Deja en salidas/:
    plan_llegadas_A013.csv        plan de llegadas por mes y segmento (escenario central)
    prevision_demanda_A013.csv    demanda esperada por mes, segmento y escenario
    tabla5_total.csv              Tabla 5 (media, p10 y p90 de 1.000 años simulados)
    tabla5_segmentos.csv          Tabla 5 por segmento (media)
"""
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))
warnings.simplefilter("ignore")

import pandas as pd

from datos import cargar, panel_segmento, macro, SEGMENTOS
from simulacion import (prevision_a013, economia_coche, z_por_segmento, situacion_inicial,
                        simular, plan_central, tabla5)

# Error del modelo medido en la prueba con A012 (notebook 02, §8)
SIGMAS = {"MSUV": 0.17, "SSUV": 0.14, "C": 0.51, "B": 0.29, "MD": 0.30, "S": 0.23, "CSUV": 0.51}
SIGMA_SESGO = 0.16
R = 1000
SEMILLA = 13


def main():
    salidas = RAIZ / "salidas"
    salidas.mkdir(exist_ok=True)

    d = cargar()
    p = panel_segmento(d)
    mac = macro(d)

    print("1/4 Previsión de A013 por escenario...")
    prev, info = prevision_a013(d, p, mac)
    pd.concat(prev, names=["escenario", "fecha"]).round(1).to_csv(salidas / "prevision_demanda_A013.csv")

    print("2/4 Política de pedidos (newsvendor) y situación de partida...")
    econ = economia_coche(d["segmentos"])
    zs = z_por_segmento(econ)
    stock0, en_camino = situacion_inicial(d)

    print("3/4 Plan de llegadas (central)...")
    plan = plan_central(prev, econ, zs, SIGMAS, SIGMA_SESGO, stock0, en_camino)
    plan["TOTAL"] = plan.sum(axis=1)
    plan.to_csv(salidas / "plan_llegadas_A013.csv")

    print(f"4/4 Simulación: {R} años x 3 escenarios x 7 segmentos...")
    res = simular(prev, econ, zs, SIGMAS, SIGMA_SESGO, stock0, en_camino, R=R, semilla=SEMILLA)
    t5 = pd.concat({e: tabla5(res, estadistico=e) for e in ["media", "p10", "p90"]}, axis=1)
    t5.round(1).to_csv(salidas / "tabla5_total.csv")
    pd.concat({s: tabla5(res, s) for s in SEGMENTOS}).round(1).to_csv(salidas / "tabla5_segmentos.csv")

    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print("\nTABLA 5 (total, media):")
    print(tabla5(res))
    print(f"\nPlan de llegadas A013 (central): {plan['TOTAL'].sum():,} coches")


if __name__ == "__main__":
    main()
