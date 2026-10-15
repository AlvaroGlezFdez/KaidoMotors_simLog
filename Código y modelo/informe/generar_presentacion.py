"""Genera la presentación (portada + 5 diapositivas, 16:9) con el estilo de la plantilla UAX.

    python run.py
    python informe/generar_presentacion.py
    (después se imprime informe/presentacion.html a PDF con Chrome/Edge headless)
"""
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datos import cargar, panel_segmento, macro, SEGMENTOS
from graficos import NOMBRES, COLORES, estilo, eje_miles, TINTA, TINTA_2, APAGADO, BANDA, SUCESO
from simulacion import (prevision_a013, economia_coche, z_por_segmento, situacion_inicial,
                        simular, plan_central, tabla5, MESES, NOMBRES_MES)
from run import SIGMAS, SIGMA_SESGO, R, SEMILLA

FIGS = RAIZ / "informe" / "figuras_presentacion"
FIGS.mkdir(parents=True, exist_ok=True)
ESC = ["Pesimista", "Central", "Optimista"]
AZUL_UAX = "#0a5ea8"
NAVY = "#14213d"
COLOR_ESC = {"Pesimista": "#d64545", "Central": AZUL_UAX, "Optimista": "#1f9d6b"}


def es(x, dec=0):
    if dec < 0:
        x, dec = round(x, dec), 0
    s = f"{x + 0.0:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def meur(x, dec=1):
    return es(x / 1e6, dec)


# --- Datos y resultados (mismos parámetros y semilla que run.py) ---------------------------
d = cargar()
p = panel_segmento(d)
mac = macro(d)
prev, info = prevision_a013(d, p, mac)
econ = economia_coche(d["segmentos"])
zs = z_por_segmento(econ)
stock0, en_camino = situacion_inicial(d)
plan = plan_central(prev, econ, zs, SIGMAS, SIGMA_SESGO, stock0, en_camino)
res = simular(prev, econ, zs, SIGMAS, SIGMA_SESGO, stock0, en_camino, R=R, semilla=SEMILLA)
t5, p10, p90 = tabla5(res), tabla5(res, estadistico="p10"), tabla5(res, estadistico="p90")

mercado = p["mercado"].unstack("segmento").asfreq("MS")[SEGMENTOS]
demanda = p["demanda"].unstack("segmento").asfreq("MS")[SEGMENTOS]
total = mercado.sum(axis=1)
real12 = demanda.loc["2012"].sum()
M = lambda k, e: t5.loc[k, e]
SEG = lambda s, k, e: res[e][s][k].mean()
cvp = {e: M("Margen de la marca, CVP (€)", e) for e in ESC}
vend_c = {s: SEG(s, "vendidos", "Central") for s in SEGMENTOS}
co2_c = sum(vend_c[s] * econ.loc[s, "prima_co2"] for s in SEGMENTOS)
co2_pes = sum(res["Pesimista"][s]["vendidos"] * econ.loc[s, "prima_co2"] for s in SEGMENTOS).mean()
resto = cvp["Central"] - cvp["Pesimista"] - co2_pes
total_plan = int(plan.to_numpy().sum())
dem_c = prev["Central"].sum().sum()
suv = sum(SEG(s, "cvp", "Central") for s in ["MSUV", "SSUV", "CSUV"]) / cvp["Central"]
m12, m13 = total.loc["2012"].sum(), info["mercado"]["Central"].sum()
dic = d["stock_mensual"].query("anio == 'A012' and mes == 12").set_index("segmento")["stock_total"]
orden = ["MSUV", "SSUV", "CSUV", "C", "B", "MD", "S"]

# --- Figuras para diapositiva (letra grande) -----------------------------------------------
estilo()
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "legend.fontsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.titlepad": 8,
                     "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.facecolor": "#ffffff"})


def guardar(fig, nombre):
    fig.savefig(FIGS / f"{nombre}.png", dpi=200)
    plt.close(fig)


# P1 · demanda limpia mensual de la gama de siempre
gama = demanda.drop(columns="CSUV").sum(axis=1)
g11, g12 = gama.loc["2011"].to_numpy(), gama.loc["2012"].to_numpy()
fig, ax = plt.subplots(figsize=(6.6, 3.5))
ax.axvspan(10.5, 12.5, color="#e8eff7", lw=0, zorder=0)
ax.plot(range(1, 13), g11, color=APAGADO, marker="o", ms=5, lw=2, label="A011")
ax.plot(range(1, 13), g12, color=AZUL_UAX, marker="o", ms=5, lw=2.4, label="A012")
ax.plot([8], [g11[7]], "o", ms=11, color=SUCESO, zorder=5)
ax.annotate("compras adelantadas\nA011", (8, g11[7]), xytext=(5.6, 2150), fontsize=9.5, color=TINTA_2,
            arrowprops=dict(arrowstyle="-", color=APAGADO, lw=0.8))
ax.text(11.5, 2330, "compras\nadelantadas A012", ha="center", va="top", fontsize=9.5, color=TINTA_2)
ax.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylim(1000, 2350)
eje_miles(ax)
ax.legend(loc="lower left", ncols=2)
ax.set_title("Demanda real mensual de la gama de siempre (coches)")
fig.tight_layout()
guardar(fig, "p1_demanda_real")

# P2 · mercado por escenario
fig, ax = plt.subplots(figsize=(6.6, 3.5))
hist = total["2011-01":"2012-12"]
ax.plot(hist.index, hist / 1000, color=TINTA, lw=1.8, label="real")
for e in ESC:
    ax.plot(MESES, info["mercado"][e] / 1000, color=COLOR_ESC[e], lw=2.2, label=e.lower())
ax.axvline(pd.Timestamp("2012-12-16"), color=APAGADO, lw=1, ls=":")
ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in (2011, 2012, 2013)], ["A011", "A012", "A013"])
ax.set_ylim(35, 105)
ax.set_title("Mercado de los 7 segmentos (miles de coches al mes)")
ax.legend(loc="lower left", ncols=4, fontsize=9.5, handlelength=1.4, columnspacing=1)
fig.tight_layout()
guardar(fig, "p2_mercado")

# P3 · plan de llegadas apilado
fig, ax = plt.subplots(figsize=(7.6, 3.6))
abajo = np.zeros(12)
for s in orden:
    ax.bar(range(12), plan[s], bottom=abajo, color=COLORES[s], width=0.72, label=NOMBRES[s],
           edgecolor="#ffffff", linewidth=0.8)
    abajo += plan[s].to_numpy()
for i, v in enumerate(abajo):
    ax.text(i, v + 45, es(v), ha="center", fontsize=8.5, color=TINTA_2)
ax.axvspan(-0.5, 1.5, color="#eeeeea", lw=0, zorder=0)
ax.text(0.5, abajo.max() * 1.14, "ya en camino", ha="center", fontsize=9.5, color=TINTA_2)
ax.set_xticks(range(12), NOMBRES_MES)
ax.set_ylim(0, abajo.max() * 1.24)
eje_miles(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=4, fontsize=9, frameon=False)
ax.set_title("Llegadas a la red por mes y segmento (coches)")
fig.tight_layout()
guardar(fig, "p3_plan")

# P4 · margen por escenario con rango
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for i, e in enumerate(ESC):
    v = cvp[e] / 1e6
    lo, hi = p10.loc["Margen de la marca, CVP (€)", e] / 1e6, p90.loc["Margen de la marca, CVP (€)", e] / 1e6
    ax.bar(i, v, color=COLOR_ESC[e], width=0.58)
    ax.plot([i, i], [lo, hi], color=TINTA, lw=1.4)
    ax.plot([i - 0.08, i + 0.08], [lo, lo], color=TINTA, lw=1.4)
    ax.plot([i - 0.08, i + 0.08], [hi, hi], color=TINTA, lw=1.4)
    ax.text(i + 0.33, v, f"{meur(cvp[e])} M€", va="center", fontsize=12, fontweight="bold")
ax.set_xticks(range(3), ESC)
ax.set_xlim(-0.5, 2.9)
ax.set_ylim(0, 140)
ax.set_ylabel("millones de €")
ax.set_title("Margen de la marca (CVP) en A013 · barra: media · línea: rango probable")
fig.tight_layout()
guardar(fig, "p4_margen")

# P5 · cascada central -> pesimista
fig, ax = plt.subplots(figsize=(6.0, 3.6))
pasos = [("Margen\ncentral", cvp["Central"], 0, AZUL_UAX), ("Prima\nde CO₂", -co2_pes, cvp["Central"], "#d64545"),
         ("Resto de riesgos\ndel pesimista", -resto, cvp["Central"] - co2_pes, "#ec835a"),
         ("Margen\npesimista", cvp["Pesimista"], 0, TINTA_2)]
for i, (n, v, base, c) in enumerate(pasos):
    b = base + v if v < 0 else base
    ax.bar(i, abs(v) / 1e6, bottom=b / 1e6, color=c, width=0.6)
    ax.text(i, (b + abs(v)) / 1e6 + 2.5, ("−" if i in (1, 2) else "") + meur(abs(v)) + " M€",
            ha="center", fontsize=11.5, fontweight="bold")
ax.set_xticks(range(4), [x[0] for x in pasos], fontsize=9.5)
ax.set_ylim(0, cvp["Central"] / 1e6 * 1.2)
ax.set_ylabel("millones de €")
ax.set_title("Del escenario central al pesimista: qué se come el margen")
fig.tight_layout()
guardar(fig, "p5_cascada")

# --- Piezas HTML -----------------------------------------------------------------------------
filas_dem = "".join(
    f"<tr><td><span class='dot' style='background:{COLORES[s]}'></span>{NOMBRES[s]}</td>"
    f"<td class='n'>{es(real12[s])}</td><td class='n'><b>{es(prev['Central'][s].sum())}</b></td>"
    f"<td class='n {'neg' if prev['Central'][s].sum() < real12[s] else 'pos'}'>"
    f"{'+' if prev['Central'][s].sum() >= real12[s] else ''}{es(100 * (prev['Central'][s].sum() / real12[s] - 1))} %</td></tr>"
    for s in orden)

filas_t5 = "".join(
    f"<tr><td>{n}</td>" + "".join(f"<td class='n'>{f(e)}</td>" for e in ESC) + "</tr>"
    for n, f in [
        ("Coches vendidos", lambda e: es(M("Coches vendidos (unidades)", e))),
        ("Coches fabricados", lambda e: es(M("Coches fabricados / llegadas (unidades)", e))),
        ("Stock medio (meses de venta)", lambda e: es(M("Stock medio en la red (meses de venta)", e), 1)),
        ("Coches &gt; 90 días a fin de año", lambda e: es(M("Coches con más de 90 días a fin de año", e))),
        ("Ventas perdidas / transfers", lambda e: f'{es(M("Ventas perdidas (coches)", e))} / {es(M("Transfers (coches)", e))}'),
        ("Coste stock + transfers (M€)", lambda e: meur(M("Coste del stock para la marca (€)", e) + M("Transfers (€ de coste)", e))),
        ("Gasto comercial (M€)", lambda e: meur(M("Gasto comercial, VM (€)", e))),
        ("<b>Margen CVP (M€)</b>", lambda e: f"<b>{meur(cvp[e])}</b>"),
    ])

seg_cvp = "".join(
    f"<div class='chip'><span class='dot' style='background:{COLORES[s]}'></span>{NOMBRES[s]} <b>{meur(SEG(s, 'cvp', 'Central'))}</b></div>"
    for s in orden)

N = 6
pie = lambda i: f"<div class='pie'><span>Kaido Motors España · Plan de producción A013</span><span>{i:02d} / {N:02d}</span></div>"
LOGO = """<div class="logo"><span class="ua">UA</span><span class="x">X</span><span class="nom">Universidad<br>Alfonso X el Sabio</span></div>"""

slides = []
slides.append(f"""<section class="slide portada">
<div class="circulo c1"></div><div class="circulo c2"></div><div class="circulo c3"></div>
<div class="barra">SIMULACIÓN DE SISTEMAS LOGÍSTICOS · CASO KAIDO MOTORS ESPAÑA</div>
<h1>Plan de producción A013</h1>
<div class="subt">Cuántos coches fabricar, en qué segmentos y cuánto se juega Kaido</div>
<div class="logo grande"><span class="ua">UA</span><span class="x">X</span><span class="nom">Universidad<br>Alfonso X el Sabio</span></div>
<div class="autor">Álvaro González Fernández · 4º Ingeniería Matemática · 15 de octubre de 2026</div>
</section>""")

slides.append(f"""<section class="slide">{LOGO}
<div class="kicker">01 · Diagnóstico de A012</div>
<h2>A012 parecía un buen año, pero la demanda real cayó</h2>
<div class="fila">
<div class="col-fig"><img src="figuras_presentacion/p1_demanda_real.png"></div>
<div class="col-txt">
<div class="tarjeta"><div class="big">+8,5 % → −4 %</div>Kaido vendió un 8,5 % más, pero sin automatriculaciones, sin el SUV compacto nuevo y sin las compras adelantadas por la normativa, <b>la demanda real de la gama de siempre cayó un 4 %</b>.</div>
<div class="tarjeta"><div class="big">−20 M€</div>Se fabricó para un mercado que crecía: la red se llenó, el <b>35 %</b> de los coches pasó de 90 días y hubo que dar descuentos: <b>460 € menos por coche</b> en A011–A012.</div>
<div class="tarjeta"><div class="big">74 %</div>del stock a 31 de diciembre eran compactos y SUV compactos, mientras el <b>SUV pequeño</b>, el 2º más vendido, estaba <b>a cero</b>.</div>
</div></div>
<div class="mensaje">Las ventas no miden la demanda: el plan de A013 se construye sobre la <b>demanda real</b> y segmento a segmento.</div>
{pie(2)}</section>""")

slides.append(f"""<section class="slide">{LOGO}
<div class="kicker">02 · Previsión de la demanda A013</div>
<h2>El mercado cae un 14 %; Kaido aguanta gracias al SUV compacto</h2>
<div class="fila">
<div class="col-fig"><img src="figuras_presentacion/p2_mercado.png">
<div class="mini">Modelo validado: al prever A012 con datos hasta A011, el error en el mercado fue <b>menor del 1 %</b> (repetir el año anterior: 4,9 %).</div></div>
<div class="col-txt">
<table><tr><th>Demanda Kaido (coches)</th><th>A012</th><th>A013</th><th>Var.</th></tr>{filas_dem}
<tr class="tot"><td>Total</td><td class="n">{es(real12.sum())}</td><td class="n">{es(dem_c)}</td><td class="n pos">+{es(100 * (dem_c / real12.sum() - 1))} %</td></tr></table>
<div class="tarjeta compacta"><b>Por qué cae el mercado</b> ({es(m12 / 1000)}.000 → {es(m13 / 1000)}.000): la <b>confianza del consumidor</b> se hundió 16 puntos en A012 y se devuelven ≈15.000 compras adelantadas. <b>Por qué Kaido sube</b>: el SUV compacto vende 12 meses en vez de 5.</div>
</div></div>
{pie(3)}</section>""")

slides.append(f"""<section class="slide">{LOGO}
<div class="kicker">03 · Propuesta</div>
<h2>Fabricar {es(total_plan)} coches, segmento a segmento</h2>
<div class="fila">
<div class="col-fig ancha"><img src="figuras_presentacion/p3_plan.png"></div>
<div class="col-txt estrecha">
<div class="tarjeta compacta"><div class="num">01</div><b>Compacto: producción parada de marzo a agosto.</b> Tiene ≈9 meses de venta entre la red y lo que viene en camino.</div>
<div class="tarjeta compacta"><div class="num">02</div><b>SUV pequeño: {es(plan.loc['Mar', 'SSUV'])} coches en marzo</b> para rellenar una red que empieza a cero.</div>
<div class="tarjeta compacta"><div class="num">03</div><b>Stock de seguridad propio por segmento:</b> quedarse sin coches cuesta 3–5 veces más que tenerlos de más. El pequeño, sin colchón: el transfer pierde dinero.</div>
</div></div>
<div class="mensaje">Plan rodante: cada mes se revisa con las ventas reales y solo se corrige el pedido que llega dentro de 3 meses.</div>
{pie(4)}</section>""")

slides.append(f"""<section class="slide">{LOGO}
<div class="kicker">04 · Resultados financieros (1.000 años simulados por escenario)</div>
<h2>{meur(cvp['Central'], 0)} M€ de margen esperado con ≈1 mes de stock en la red</h2>
<div class="fila">
<div class="col-fig"><img src="figuras_presentacion/p4_margen.png"></div>
<div class="col-txt">
<table class="t5"><tr><th></th><th>Pesimista</th><th>Central</th><th>Optimista</th></tr>{filas_t5}</table>
<div class="mini" style="margin-top:10px"><b>Margen central por segmento (M€):</b></div>
<div class="chips">{seg_cvp}</div>
</div></div>
<div class="mensaje">Los tres SUV generan el <b>{es(100 * suv)} %</b> del margen. El optimista se cubre con transfers rentables; el pesimista se hunde por la prima de CO₂.</div>
{pie(5)}</section>""")

slides.append(f"""<section class="slide">{LOGO}
<div class="kicker">05 · Riesgos y recomendaciones</div>
<h2>El mayor riesgo es la prima de CO₂, no la demanda</h2>
<div class="fila">
<div class="col-fig"><img src="figuras_presentacion/p5_cascada.png">
<div class="mini">Con las ventas del escenario central, la prima costaría <b>≈{meur(co2_c, 0)} M€</b> (≈3.000 € por coche, el {es(100 * co2_c / cvp['Central'])} % del margen). Equivocarse en el stock de seguridad mueve menos de 1 M€.</div></div>
<div class="col-txt">
<div class="reco"><span>01</span><div><b>Fabricar {es(total_plan)} coches</b> y mantener ≈1 mes de stock en la red.</div></div>
<div class="reco"><span>02</span><div><b>Empujar los SUV</b>, reducir el compacto y frenar el pequeño.</div></div>
<div class="reco"><span>03</span><div><b>Revisar cada mes</b> la cuota de Kaido, la confianza del consumidor y el stock de más de 90 días.</div></div>
<div class="reco"><span>04</span><div><b>Reducir las automatriculaciones</b> (≈11 % de las ventas).</div></div>
<div class="reco"><span>05</span><div><b>Pedir a la matriz la decisión sobre el CO₂</b> antes de marzo y preparar el plan alternativo.</div></div>
</div></div>
<div class="mensaje">Pedimos al comité aprobar el plan de {es(total_plan)} coches y elevar a la matriz la decisión sobre la prima de CO₂.</div>
{pie(6)}</section>""")

html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Kaido Motors · Plan de producción A013</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
@page {{ size: 1280px 720px; margin: 0; }}
:root {{ --azul:{AZUL_UAX}; --navy:{NAVY}; --gris:#5b6472; --claro:#e8eff7; --borde:#d3e0ee; --grid:#e3e6ea; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:"IBM Plex Sans","Segoe UI",sans-serif; color:var(--navy); }}
.slide {{ width:1280px; height:720px; position:relative; overflow:hidden; background:#fff; padding:44px 64px 0 64px; break-after:page; }}
.logo {{ position:absolute; top:30px; right:56px; display:flex; align-items:center; gap:6px; }}
.logo .ua, .logo .x {{ font-family:"DM Sans","Segoe UI",sans-serif; font-weight:800; font-size:30px; letter-spacing:-1px; line-height:1; }}
.logo .ua {{ color:#1b1b1b; }} .logo .x {{ color:var(--azul); margin-left:-6px; }}
.logo .nom {{ font-size:9.5px; line-height:1.1; color:#1b1b1b; font-weight:500; margin-left:4px; }}
.logo.grande {{ position:static; margin-top:56px; }}
.logo.grande .ua, .logo.grande .x {{ font-size:58px; }} .logo.grande .nom {{ font-size:15px; }}
.kicker {{ color:var(--azul); font-weight:700; font-size:15px; letter-spacing:1.2px; text-transform:uppercase; margin-top:6px; }}
h2 {{ font-family:"DM Sans","Segoe UI",sans-serif; font-size:36px; font-weight:800; margin:8px 0 22px 0; letter-spacing:-0.6px; line-height:1.12; max-width:1000px; }}
.fila {{ display:flex; gap:30px; align-items:flex-start; }}
.col-fig {{ flex:1.15; }} .col-fig.ancha {{ flex:1.6; }}
.col-txt {{ flex:1; display:flex; flex-direction:column; gap:12px; }} .col-txt.estrecha {{ flex:0.9; }}
img {{ width:100%; display:block; border-radius:10px; }}
.tarjeta {{ background:var(--claro); border:1px solid var(--borde); border-radius:14px; padding:13px 18px; font-size:15.5px; line-height:1.38; }}
.tarjeta.compacta {{ font-size:15px; padding:11px 16px; }}
.tarjeta .big {{ font-family:"DM Sans",sans-serif; color:var(--azul); font-size:27px; font-weight:800; line-height:1.1; margin-bottom:3px; }}
.tarjeta .num {{ font-family:"DM Sans",sans-serif; color:var(--azul); font-size:24px; font-weight:800; float:left; margin:-2px 12px 0 0; }}
.mensaje {{ position:absolute; left:64px; right:64px; bottom:58px; background:var(--azul); color:#fff; border-radius:24px; padding:12px 24px; font-size:17px; }}
.mini {{ font-size:13.5px; color:var(--gris); margin-top:8px; line-height:1.35; }}
table {{ border-collapse:collapse; width:100%; font-size:15px; font-variant-numeric:tabular-nums; }}
th {{ text-align:right; color:var(--gris); font-weight:600; font-size:13.5px; border-bottom:2px solid var(--navy); padding:5px 8px; }}
th:first-child {{ text-align:left; }}
td {{ padding:4.5px 8px; border-bottom:1px solid var(--grid); }}
td.n {{ text-align:right; white-space:nowrap; }} td.neg {{ color:#c23b3b; }} td.pos {{ color:#1f7a52; }}
tr.tot td {{ font-weight:700; border-top:2px solid var(--navy); }}
table.t5 td {{ padding:4px 8px; font-size:14.5px; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }}
.chip {{ background:var(--claro); border-radius:14px; padding:3px 10px; font-size:13px; }}
.reco {{ background:var(--claro); border:1px solid var(--borde); border-radius:14px; padding:10px 16px; font-size:15.5px; display:flex; gap:14px; align-items:center; line-height:1.3; }}
.reco span {{ font-family:"DM Sans",sans-serif; color:var(--azul); font-weight:800; font-size:22px; }}
.pie {{ position:absolute; left:64px; right:64px; bottom:18px; display:flex; justify-content:space-between; color:var(--gris); font-size:13px; }}
.portada {{ padding:0 96px; display:flex; flex-direction:column; justify-content:center; }}
.portada .barra {{ background:var(--azul); color:#fff; font-weight:700; font-size:16px; letter-spacing:1.4px; border-radius:24px; padding:12px 26px; width:fit-content; margin-bottom:26px; position:relative; z-index:2; }}
.portada h1 {{ font-family:"DM Sans","Segoe UI",sans-serif; font-size:76px; font-weight:800; margin:0; letter-spacing:-2px; position:relative; z-index:2; }}
.portada .subt {{ font-size:26px; color:var(--gris); margin-top:14px; max-width:820px; position:relative; z-index:2; }}
.portada .autor {{ position:absolute; bottom:40px; left:96px; color:var(--gris); font-size:16px; }}
.circulo {{ position:absolute; border-radius:50%; }}
.c1 {{ width:560px; height:560px; background:var(--claro); right:-130px; top:-150px; }}
.c2 {{ width:220px; height:220px; border:2px solid #a9c6e3; right:60px; top:-40px; }}
.c3 {{ width:300px; height:300px; border:2px solid #a9c6e3; left:-120px; bottom:-170px; }}
</style></head><body>
{''.join(slides)}
</body></html>"""

salida = RAIZ / "informe" / "presentacion.html"
salida.write_text(html, encoding="utf-8")
print("ok", salida, len(slides), "diapositivas")
