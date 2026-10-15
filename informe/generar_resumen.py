"""Genera el resumen ejecutivo (2 páginas A4) a partir del pipeline.

    python run.py
    python informe/generar_resumen.py
    (después se imprime informe/resumen_ejecutivo.html a PDF con Chrome/Edge headless)
"""
import sys
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))
warnings.simplefilter("ignore")

import pandas as pd
import matplotlib.pyplot as plt

from datos import cargar, panel_segmento, macro, SEGMENTOS
from graficos import estilo, NOMBRES, COLORES, AZUL, ROJO, TINTA, TINTA_2, APAGADO, BANDA, SUCESO, eje_miles
from simulacion import (prevision_a013, economia_coche, z_por_segmento, situacion_inicial,
                        simular, plan_central, tabla5, MESES, NOMBRES_MES)
from run import SIGMAS, SIGMA_SESGO, R, SEMILLA

FIGS = RAIZ / "informe" / "figuras_resumen"
FIGS.mkdir(parents=True, exist_ok=True)
ESC = ["Pesimista", "Central", "Optimista"]
COLOR_ESC = {"Pesimista": "#e34948", "Central": "#2a78d6", "Optimista": "#1baf7a"}


def es(x, dec=0):
    if dec < 0:                       # redondear a decenas, centenas...
        x, dec = round(x, dec), 0
    s = f"{x + 0.0:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def meur(x, dec=1):
    return es(x / 1e6, dec)


# --- Datos y resultados (mismos parámetros y semilla que run.py) ---------------------------
d = cargar()
p = panel_segmento(d)
mac = macro(d)
seg_csv = d["segmentos"]
prev, info = prevision_a013(d, p, mac)
econ = economia_coche(seg_csv)
zs = z_por_segmento(econ)
stock0, en_camino = situacion_inicial(d)
plan = plan_central(prev, econ, zs, SIGMAS, SIGMA_SESGO, stock0, en_camino)
res = simular(prev, econ, zs, SIGMAS, SIGMA_SESGO, stock0, en_camino, R=R, semilla=SEMILLA)
t5, p10, p90 = tabla5(res), tabla5(res, estadistico="p10"), tabla5(res, estadistico="p90")

mercado = p["mercado"].unstack("segmento").asfreq("MS")[SEGMENTOS]
demanda = p["demanda"].unstack("segmento").asfreq("MS")[SEGMENTOS]
ventas = p["ventas"].unstack("segmento").asfreq("MS")[SEGMENTOS]
automat = p["automat"].unstack("segmento").asfreq("MS")[SEGMENTOS]
total = mercado.sum(axis=1)
real12 = demanda.loc["2012"].sum()

M = lambda k, e: t5.loc[k, e]
SEG = lambda s, k, e: res[e][s][k].mean()
cvp = {e: M("Margen de la marca, CVP (€)", e) for e in ESC}
co2_pes = sum(res["Pesimista"][s]["vendidos"] * econ.loc[s, "prima_co2"] for s in SEGMENTOS).mean()
resto = cvp["Central"] - cvp["Pesimista"] - co2_pes
vend_c = {s: SEG(s, "vendidos", "Central") for s in SEGMENTOS}
co2_c = sum(vend_c[s] * econ.loc[s, "prima_co2"] for s in SEGMENTOS)
ingreso_c = sum(vend_c[s] * (seg_csv.loc[s, "pff_medio"] - seg_csv.loc[s, "margen_bruto_conc_eur"]) for s in SEGMENTOS)
fabricar_c = sum(vend_c[s] * econ.loc[s, "coste_fabricar"] for s in SEGMENTOS)
vm_c = M("Gasto comercial, VM (€)", "Central")
total_plan = int(plan.to_numpy().sum())
dem_c = prev["Central"].sum().sum()
gama_13 = prev["Central"].drop(columns="CSUV").sum().sum()
gama_12 = real12.drop("CSUV").sum()
suv = sum(SEG(s, "cvp", "Central") for s in ["MSUV", "SSUV", "CSUV"]) / cvp["Central"]
m12 = total.loc["2012"].sum()
m13 = info["mercado"]["Central"].sum()
adel = info["adelantados"]
efecto_eco = 100 * ((m13 + adel) / (m12 - adel) - 1)
conf_ene, conf_dic = mac.loc["2012-01-01", "confianza_consumidor"], mac.loc["2012-12-01", "confianza_consumidor"]
dic = d["stock_mensual"].query("anio == 'A012' and mes == 12").set_index("segmento")["stock_total"]
pct_c_csuv = 100 * (dic["C"] + dic["CSUV"]) / dic.sum()
rango_c = f"{meur(p10.loc['Margen de la marca, CVP (€)', 'Central'], 0)}–{meur(p90.loc['Margen de la marca, CVP (€)', 'Central'], 0)}"

anual = pd.DataFrame({
    "Ventas (lo que reporta Kaido)": ventas.groupby(ventas.index.year).sum().sum(axis=1),
    "− Automatriculaciones": automat.groupby(automat.index.year).sum().sum(axis=1),
    "− SUV compacto (nuevo)": demanda["CSUV"].groupby(demanda.index.year).sum(),
    "= Demanda limpia, gama de siempre": demanda.drop(columns="CSUV").groupby(demanda.index.year).sum().sum(axis=1),
}).loc[2010:2012]

# --- Figuras compactas -------------------------------------------------------------------
estilo()
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlepad": 4})


def guardar(fig, nombre):
    fig.savefig(FIGS / f"{nombre}.png", dpi=220)
    plt.close(fig)


# Demanda limpia mensual de la gama de siempre, A011 frente a A012
gama = demanda.drop(columns="CSUV").sum(axis=1)
g11, g12 = gama.loc["2011"].to_numpy(), gama.loc["2012"].to_numpy()
fig, ax = plt.subplots(figsize=(4.5, 1.55))
ax.axvspan(10.5, 12.5, color=BANDA, lw=0, zorder=0)
ax.plot(range(1, 13), g11, color=APAGADO, marker="o", ms=3, lw=1.5, label="A011")
ax.plot(range(1, 13), g12, color=AZUL, marker="o", ms=3, lw=1.8, label="A012")
ax.plot([8], [g11[7]], "o", ms=7, color=SUCESO, zorder=5)
ax.annotate("adelanto A011", (8, g11[7]), xytext=(8.25, 2120), fontsize=6.8, color=TINTA_2, ha="left")
ax.text(11.5, 2290, "adelanto A012", ha="center", fontsize=6.8, color=TINTA_2)
ax.set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax.set_ylim(1000, 2400)
eje_miles(ax)
ax.legend(loc="lower left", ncols=2, fontsize=6.8)
ax.set_title("Demanda limpia mensual de la gama de siempre (coches)")
fig.tight_layout()
guardar(fig, "f6_gama_mensual")

# Mercado: historia y A013 por escenario
fig, ax = plt.subplots(figsize=(4.5, 1.6))
hist = total["2011-01":"2012-12"]
ax.plot(hist.index, hist / 1000, color=TINTA, lw=1.4, label="real")
for e in ESC:
    ax.plot(MESES, info["mercado"][e] / 1000, color=COLOR_ESC[e], lw=1.6, label=e.lower())
ax.axvline(pd.Timestamp("2012-12-16"), color=APAGADO, lw=0.8, ls=":")
ax.set_xticks([pd.Timestamp(f"{a}-01-01") for a in (2011, 2012, 2013)], ["A011", "A012", "A013"])
ax.set_ylim(35, 105)
ax.set_title("Mercado de los 7 segmentos (miles de coches/mes)")
ax.legend(loc="lower left", ncols=4, fontsize=6.5, handlelength=1.2, columnspacing=0.8)
fig.tight_layout()
guardar(fig, "f2_mercado")

# Distribución del margen
fig, ax = plt.subplots(figsize=(4.4, 1.7))
for e in ESC:
    v = sum(res[e][s]["cvp"] for s in SEGMENTOS) / 1e6
    ax.hist(v, bins=40, color=COLOR_ESC[e], alpha=0.8, label=f"{e}: {es(v.mean(), 0)} M€")
ax.set_xlabel("Margen CVP de A013 (M€)", fontsize=7)
ax.set_ylabel("años simulados", fontsize=7)
ax.grid(False)
ax.legend(loc="upper left", fontsize=6.8)
ax.set_title("Margen en 1.000 años simulados por escenario")
fig.tight_layout()
guardar(fig, "f4_margen")

# Cascada central -> pesimista
fig, ax = plt.subplots(figsize=(4.4, 1.75))
pasos = [("CVP\ncentral", cvp["Central"], 0, AZUL), ("Prima\nde CO₂", -co2_pes, cvp["Central"], ROJO),
         ("Resto del\npesimista*", -resto, cvp["Central"] - co2_pes, "#ec835a"),
         ("CVP\npesimista", cvp["Pesimista"], 0, TINTA_2)]
for i, (n, v, base, c) in enumerate(pasos):
    b = base + v if v < 0 else base
    ax.bar(i, abs(v) / 1e6, bottom=b / 1e6, color=c, width=0.62)
    ax.text(i, (b + abs(v)) / 1e6 + 2.5, ("−" if i in (1, 2) else "") + meur(abs(v)) + " M€",
            ha="center", fontsize=7.5, fontweight="bold")
ax.set_xticks(range(4), [x[0] for x in pasos], fontsize=7)
ax.set_ylim(0, cvp["Central"] / 1e6 * 1.22)
ax.set_title("Del central al pesimista: qué se come el margen (M€)")
fig.tight_layout()
guardar(fig, "f5_cascada")

# --- Tablas HTML -------------------------------------------------------------------------
cab_anual = "".join(f"<th>A{a - 2000:03d}</th>" for a in anual.index)
filas_anual = "".join(
    f"<tr class='{'tot' if c.startswith('=') else ''}'><td>{c}</td>"
    + "".join(f"<td class='n'>{es(anual.loc[a, c])}</td>" for a in anual.index) + "</tr>"
    for c in anual.columns)

orden = ["MSUV", "SSUV", "CSUV", "C", "B", "MD", "S"]
cab_plan = "".join(f"<th>{m}</th>" for m in NOMBRES_MES) + "<th>Total</th>"
filas_plan = ""
for s in orden + ["TOTAL"]:
    serie = plan.sum(axis=1) if s == "TOTAL" else plan[s]
    nombre = "Total" if s == "TOTAL" else f"<span class='dot' style='background:{COLORES[s]}'></span>{NOMBRES[s]}"
    celdas = "".join(
        f"<td class='n{' camino' if i < 2 else ''}{' cero' if serie.iloc[i] == 0 else ''}'>{es(serie.iloc[i])}</td>"
        for i in range(12))
    filas_plan += (f"<tr class='{'tot' if s == 'TOTAL' else ''}'><td>{nombre}</td>{celdas}"
                   f"<td class='n'><b>{es(serie.sum())}</b></td></tr>")

filas_t5_def = [
    ("Coches vendidos <span class='sub'>(incluye transfers)</span>", lambda e: es(M("Coches vendidos (unidades)", e))),
    ("Coches fabricados <span class='sub'>(llegadas a la red)</span>", lambda e: es(M("Coches fabricados / llegadas (unidades)", e))),
    ("Stock medio: coches · meses de venta",
     lambda e: f'{es(M("Stock medio en la red (coches)", e))} · {es(M("Stock medio en la red (meses de venta)", e), 1)}'),
    ("Coches de más de 90 días a fin de año", lambda e: es(M("Coches con más de 90 días a fin de año", e))),
    ("Coste del stock para la marca (M€)", lambda e: meur(M("Coste del stock para la marca (€)", e))),
    ("Ventas perdidas: coches · M€",
     lambda e: f'{es(M("Ventas perdidas (coches)", e))} · {meur(M("Ventas perdidas (€ de CVP)", e))}'),
    ("Transfers: coches · M€",
     lambda e: f'{es(M("Transfers (coches)", e))} · {meur(M("Transfers (€ de coste)", e))}'),
    ("Gasto comercial, VM (M€)", lambda e: meur(M("Gasto comercial, VM (€)", e))),
    ("<b>Margen de la marca, CVP (M€)</b>", lambda e: f'<b>{meur(cvp[e])}</b>'),
    ("<span class='sub'>rango probable (10 % peores – 10 % mejores)</span>",
     lambda e: f'<span class="sub">{meur(p10.loc["Margen de la marca, CVP (€)", e], 0)}–{meur(p90.loc["Margen de la marca, CVP (€)", e], 0)}</span>'),
]
filas_t5 = "".join(f"<tr><td>{n}</td>" + "".join(f"<td class='n'>{f(e)}</td>" for e in ESC) + "</tr>"
                   for n, f in filas_t5_def)

decision = {"MSUV": ("Empujar", "e", "El que más deja (≈5.700 €/coche); aguanta con CO₂."),
            "SSUV": ("Proteger", "e", "2º en margen; empieza sin coches: rellenar en marzo."),
            "CSUV": ("Empujar", "e", "Primer año completo (+3.200 coches); rentable con CO₂."),
            "MD": ("Mantener", "m", "Poco volumen, buen margen por coche."),
            "S": ("Mantener", "m", "Nicho (65 % de cuota); con CO₂ deja ≈0."),
            "C": ("Reducir", "f", "≈9 meses de stock: parar mar–ago; pierde con CO₂."),
            "B": ("Frenar", "f", "Deja ≈0; con CO₂ pierde 2.600 €/coche.")}
cmax = max(SEG(s, "cvp", "Central") for s in SEGMENTOS)
filas_seg = ""
for s in orden:
    cc, cp = SEG(s, "cvp", "Central"), SEG(s, "cvp", "Pesimista")
    dec_, clase, motivo = decision[s]
    ancho = max(cc, 0) / cmax * 66
    filas_seg += (
        f"<tr><td><span class='dot' style='background:{COLORES[s]}'></span>{NOMBRES[s]}</td>"
        f"<td class='n'>{es(real12[s])}</td><td class='n'><b>{es(vend_c[s])}</b></td>"
        f"<td class='n'>{es(SEG(s, 'meses_venta', 'Central'), 1)}</td>"
        f"<td class='barra'><span style='width:{ancho:.0f}%;background:{COLORES[s]}'></span><em>{meur(cc)}</em></td>"
        f"<td class='n{' neg' if cp < 0 else ''}'>{meur(cp)}</td>"
        f"<td><span class='tag {clase}'>{dec_}</span></td><td class='motivo'>{motivo}</td></tr>")

html = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Kaido Motors · Resumen ejecutivo A013</title>
<style>
@page {{ size: A4; margin: 8mm 10mm 7mm 10mm; }}
:root {{ --ink:#0b0b0b; --ink2:#52514e; --mut:#898781; --grid:#e1e0d9; --azul:#2a78d6; --azuld:#1c5cab;
        --rojo:#d03b3b; --fondo:#f3f6fa; --leer:#fbf8ef; }}
* {{ box-sizing:border-box; }}
body {{ font-family:"Segoe UI", system-ui, sans-serif; color:var(--ink); font-size:7.9pt; line-height:1.28; margin:0; }}
.cabecera {{ display:flex; justify-content:space-between; align-items:flex-end; border-bottom:2.5px solid var(--azul);
             padding-bottom:1.2mm; margin-bottom:2mm; }}
h1 {{ font-size:14.5pt; margin:0; letter-spacing:-0.3px; white-space:nowrap; }}
.meta {{ color:var(--ink2); font-size:7.2pt; text-align:right; line-height:1.3; white-space:nowrap; }}
h2 {{ font-size:9pt; margin:2.3mm 0 1mm 0; color:var(--azuld); }}
h2 .num {{ display:inline-block; background:var(--azul); color:#fff; border-radius:50%; width:4.4mm; height:4.4mm;
           text-align:center; line-height:4.4mm; font-size:7.3pt; margin-right:1.4mm; }}
.corta {{ background:var(--fondo); border-left:3.5px solid var(--azul); padding:1.6mm 2.6mm; font-size:8.4pt; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.8mm; margin-top:1.8mm; }}
.kpi {{ border:1px solid var(--grid); border-top:3px solid var(--azul); padding:1.2mm 2mm; }}
.kpi.r {{ border-top-color:var(--rojo); }} .kpi.r .v {{ color:var(--rojo); }}
.kpi .v {{ font-size:13pt; font-weight:800; line-height:1.1; }}
.kpi .t {{ font-weight:700; font-size:7.4pt; margin:0.3mm 0 0.4mm 0; }}
.kpi .l {{ color:var(--ink2); font-size:6.8pt; line-height:1.25; }}
.dos {{ display:grid; grid-template-columns:1fr 1fr; gap:3.5mm; }}
.dos.izq {{ grid-template-columns:1.15fr 1fr; }}
img {{ width:100%; display:block; }}
ul {{ margin:0.6mm 0 0 0; padding-left:3.8mm; }} li {{ margin:0.45mm 0; }}
table {{ border-collapse:collapse; width:100%; font-size:7.5pt; font-variant-numeric:tabular-nums; }}
th {{ text-align:right; font-weight:600; color:var(--ink2); border-bottom:1.2px solid var(--ink2); padding:0.55mm 1mm; font-size:7pt; }}
th:first-child {{ text-align:left; }}
td {{ padding:0.5mm 1mm; border-bottom:1px solid var(--grid); }}
td:first-child {{ white-space:nowrap; }}
td.n {{ text-align:right; white-space:nowrap; }} td.neg {{ color:var(--rojo); }}
td.camino {{ color:var(--mut); background:#f1f0eb; }} td.cero {{ color:var(--rojo); font-weight:600; }}
tr.tot td {{ font-weight:700; border-top:1.2px solid var(--ink2); }}
.sub {{ color:var(--mut); font-size:6.6pt; font-weight:400; }}
.dot {{ display:inline-block; width:6.5px; height:6.5px; border-radius:50%; margin-right:3px; }}
td.barra {{ position:relative; width:25mm; }}
td.barra span {{ display:block; height:3mm; border-radius:0 1.5px 1.5px 0; opacity:0.85; }}
td.barra em {{ position:absolute; right:1mm; top:0.5mm; font-style:normal; font-weight:700; }}
td.motivo {{ color:var(--ink2); font-size:6.9pt; white-space:normal; }}
.tag {{ font-size:6.7pt; padding:0.1mm 1.5mm; border-radius:2mm; font-weight:700; white-space:nowrap; }}
.tag.e {{ background:#e3efff; color:#1c5cab; }} .tag.m {{ background:#eeede8; color:#52514e; }} .tag.f {{ background:#fde8e8; color:#a52a2a; }}
.nota {{ color:var(--mut); font-size:6.6pt; margin-top:0.6mm; line-height:1.22; }}
.leer {{ background:var(--leer); border-left:3px solid #eda100; padding:1.1mm 2.2mm; margin:1.2mm 0 0 0; font-size:7.3pt; }}
.leer b.k {{ color:#8a5a00; }}
table.si td {{ vertical-align:top; padding:0.8mm 1.2mm; font-size:7.3pt; white-space:normal; }}
table.si td:first-child {{ font-weight:700; color:var(--azuld); width:30mm; }}
table.si td.acc {{ width:52mm; }}
table.si td.pq {{ color:var(--ink2); }}
.reco {{ background:var(--fondo); border-left:3.5px solid var(--azul); padding:1.6mm 2.6mm; }}
.reco ol {{ margin:0; padding-left:4.2mm; }} .reco li {{ margin:0.6mm 0; }}
.salto {{ break-before:page; }}
.pie {{ color:var(--mut); font-size:6.6pt; border-top:1px solid var(--grid); margin-top:1.8mm; padding-top:0.8mm; line-height:1.25; }}
</style></head><body>

<div class="cabecera">
<div><h1>Kaido Motors España · Plan de producción A013</h1>
<div style="color:var(--ink2);font-size:8.3pt">Resumen ejecutivo para el comité de dirección</div></div>
<div class="meta">Simulación de Sistemas Logísticos · UAX<br>Álvaro González Fernández · 15 de octubre de 2026</div>
</div>

<div class="corta"><b>La respuesta corta:</b> A013 será un año de caída. Proponemos fabricar <b>{es(total_plan)} coches</b>, concentrar el esfuerzo en los <b>SUV</b> y prepararse para la <b>prima de CO₂</b>, el mayor riesgo del año. Recomendaciones detalladas en el §8.</div>

<div class="kpis">
<div class="kpi"><div class="v">{es(total_plan)}</div><div class="t">coches a fabricar en A013</div>
<div class="l">Llegadas a la red en el escenario central: un {es(100 * (1 - total_plan / dem_c))} % menos que la demanda prevista, para vender antes el stock que sobra.</div></div>
<div class="kpi"><div class="v">{meur(cvp['Central'])} M€</div><div class="t">margen esperado (CVP)</div>
<div class="l">No es facturación: ingresos ≈{meur(ingreso_c, 0)} − fabricar y traer ≈{meur(fabricar_c, 0)} − gasto comercial ≈{meur(vm_c, 0)} M€. Rango probable {rango_c} M€.</div></div>
<div class="kpi"><div class="v">−14 %</div><div class="t">mercado en A013</div>
<div class="l">De {es(m12 / 1000)}.000 a {es(m13 / 1000)}.000 coches (nuestro modelo, economía como a final de A012): −{es(-efecto_eco)} puntos por la economía y −3 por el adelanto.</div></div>
<div class="kpi r"><div class="v">−{meur(co2_c, 0)} M€</div><div class="t">si llega la prima de CO₂</div>
<div class="l">Multa europea por emisiones que la matriz puede repartir entre países: ≈{es(co2_c / sum(vend_c.values()), -2)} € por coche, el {es(100 * co2_c / cvp['Central'])} % del margen.</div></div>
</div>

<h2><span class="num">1</span>A012: Kaido vendió más, pero los clientes no compraron más</h2>
<div class="dos">
<div>
<table><tr><th>Coches</th>{cab_anual}</tr>{filas_anual}</table>
<ul>
<li>Las ventas suben un <b>8,5 %</b>, pero sin automatriculaciones (coches que se matricula el concesionario para cumplir objetivos) y sin el SUV compacto, lanzado en agosto, <b>la gama de siempre queda plana (−0,1 %)</b>.</li>
<li>En <b>agosto de A011</b> y en <b>nov–dic de A012</b> los clientes <b>adelantaron compras</b> por cambios de normativa: ≈800 coches de Kaido (≈15.000 en el mercado) que son demanda de A013. <b>Sin ellos, la gama de siempre cayó ≈4 %.</b></li>
</ul>
</div>
<div>
<img src="figuras_resumen/f6_gama_mensual.png">
<div class="leer"><b class="k">El stock estaba en los segmentos equivocados.</b> A 31-dic, el {es(pct_c_csuv)} % de los {es(dic.sum())} coches de la red eran compactos o SUV compactos, y el <b>SUV pequeño (2º en ventas) tenía 0</b>. Por eso el plan de A013 es tan distinto por segmento.</div>
</div>
</div>

<h2><span class="num">2</span>A013: el mercado cae un 14 %, pero Kaido aguanta</h2>
<div class="dos">
<div>
<img src="figuras_resumen/f2_mercado.png">
<div class="leer"><b class="k">¿Qué es la confianza del consumidor?</b> Un índice mensual de encuestas a hogares sobre cómo ven la economía. Importa porque un coche es una compra grande que <b>se puede aplazar</b>: si la gente es pesimista, aguanta con el viejo. Se adelanta al mercado unos meses.</div>
</div>
<div>
<ul>
<li><b>Por qué cae:</b> la confianza se hundió en A012 (de {es(conf_ene)} a {es(conf_dic)} puntos) y, según el modelo, <b>cada punto vale ≈0,9 % del mercado</b> con ≈3 meses de retraso. Además se "devuelven" los coches adelantados.</li>
<li><b>Variables elegidas: confianza y paro</b> (dato de hace 3 meses): son las que mejor acompañan al mercado (correlación 0,61 y −0,31), se conocen al decidir el pedido y mueven los escenarios del caso.</li>
<li><b>Por qué Kaido aguanta:</b> su demanda prevista sube de {es(real12.sum())} a <b>{es(dem_c)}</b> porque el <b>SUV compacto</b> vende el año entero (≈{es(prev['Central']['CSUV'].sum(), -2)}). Sin él, la gama de siempre <b>cae un {es(100 * (1 - gama_13 / gama_12))} %</b>.</li>
<li><b>¿Es fiable?</b> Al prever A012 con datos hasta A011, el modelo de mercado falló <b>menos de un 1 %</b> (repetir el año anterior, 4,9 %). Lo incierto es la cuota de Kaido (≈30 % de error mensual).</li>
</ul>
</div>
</div>

<h2><span class="num">3</span>El plan de partida: {es(total_plan)} coches, segmento a segmento</h2>
<table><tr><th>Llegadas a la red</th>{cab_plan}</tr>{filas_plan}</table>
<div class="leer"><b class="k">Cómo leerlo.</b> Coches que llegarían cada mes <b>si el año sale como prevemos</b>. Ene–feb ya vienen en barco. Desde marzo es <b>un plan de partida, no fijo</b>: cada mes se recalcula solo el pedido de dentro de 3 meses con <i>pedido = demanda prevista + colchón − stock previsto</i>. Sirve para dar volumen a la matriz y comparar lo real con lo previsto.</div>
<ul>
<li><b>Compacto, 0 coches de marzo a agosto:</b> tiene {es(dic['C'])} en la red y {es(2 * en_camino[SEGMENTOS.index('C')])} en camino, ≈9 meses de venta. <b>SUV pequeño, {es(plan.loc['Mar', 'SSUV'])} en marzo</b> (el triple de lo normal): empieza a cero y en enero–febrero tirará de transfers. <b>SUV medio, solo {es(plan.loc['Mar', 'MSUV'])} en marzo:</b> vienen {es(2 * en_camino[SEGMENTOS.index('MSUV')])} en camino.</li>
<li>El pico de junio y el valle de agosto siguen la estacionalidad. Se fabrican {es(total_plan)} para una demanda de {es(dem_c)}: el resto sale del stock actual y de los transfers.</li>
</ul>

<div class="pie"><b>Cómo lo hemos calculado:</b> ① demanda real = ventas − automatriculaciones, con adelantos y meses sin coches marcados; ② previsión = mercado total (modelo de series temporales con calendario, ayudas, IVA, confianza y paro, validado prediciendo A012) × peso de cada segmento × cuota de Kaido (media de 12 meses); ③ plan mensual con pedido(t+3) = demanda prevista + colchón − stock previsto; ④ simulación de 1.000 años por escenario. <b>Limitaciones:</b> cada segmento se trata como un solo coche; transfers siempre disponibles; coches en camino en ene–feb estimados; validado con un solo año. Código: GitHub AlvaroGlezFdez/KaidoMotors_simLog (<i>python run.py</i>).</div>

<div class="salto"></div>

<h2><span class="num">4</span>Qué pasa con el plan si el año sale distinto (Tabla 5)</h2>
<div class="dos izq">
<div>
<table><tr><th>Media de 1.000 años simulados</th><th>Pesimista</th><th>Central</th><th>Optimista</th></tr>{filas_t5}</table>
</div>
<div>
<img src="figuras_resumen/f4_margen.png">
<div class="leer"><b class="k">Cómo leerlo.</b> Como la demanda real nunca es la prevista, <b>jugamos el año 1.000 veces</b> con desviaciones al azar (tan grandes como las de nuestro modelo en A012), aplicando cada mes la regla del pedido. Las tres columnas usan <b>el mismo plan</b> en tres "mundos" (Tabla 4 del caso).</div>
</div>
</div>
<ul>
<li><b>Central:</b> {meur(cvp['Central'])} M€ con ≈1 mes de stock y casi ningún coche viejo a fin de año (en dic-A012 eran el 35 % de la red).
<b>Optimista</b> (paro −1, plan de ayuda desde julio, coches nuevos que venden más): se cubre con ≈{es(round(M('Transfers (coches)', 'Optimista'), -2))} transfers, que dejan dinero → {meur(cvp['Optimista'])} M€.
<b>Pesimista</b> (paro +2, caída de 6 meses, CO₂, arancel, lanzamiento retrasado, un mes sin SUV medio): {meur(cvp['Pesimista'])} M€, <b>y no por el plan</b> (§6).</li>
</ul>

<h2><span class="num">5</span>Por segmento: qué empujar y qué frenar</h2>
<table>
<tr><th>Segmento</th><th>Demanda A012</th><th>Ventas A013</th><th>Stock (meses)</th><th style="text-align:left">Margen central M€</th><th>Margen pesim. M€</th><th style="text-align:left">Decisión</th><th style="text-align:left">Por qué</th></tr>
{filas_seg}
</table>
<div class="leer"><b class="k">Cómo leerlo.</b> La barra muestra <b>quién gana el dinero</b> (los tres SUV, el {es(100 * suv)} %); la columna del pesimista, <b>quién pierde si llega el CO₂</b> (en rojo). Se empuja lo que gana en ambos casos y se frena lo que apenas gana hoy y pierde con el CO₂.</div>

<div class="dos">
<div>
<h2><span class="num">6</span>Cuánto cuesta equivocarse</h2>
<img src="figuras_resumen/f5_cascada.png">
<div class="nota">* Paro +2, caída tras el adelanto de 6 meses, lanzamiento retrasado, arancel +3 % y un mes sin SUV medio.</div>
</div>
<div>
<h2>&nbsp;</h2>
<ul>
<li><b>La prima de CO₂ se lleva {meur(co2_pes)} M€</b>, más del doble que todos los demás riesgos del pesimista juntos ({meur(resto)} M€).</li>
<li><b>Quedarse sin coches cuesta 3–5 veces más que tener de más:</b> un transfer cuesta el 20 % del beneficio bruto (≈1.000–1.600 €) y un coche que sobra 3 meses, 200–450 €. Por eso cada segmento tiene su propio <b>stock de seguridad</b>.</li>
<li><b>El pequeño no lleva colchón:</b> un transfer le hace perder dinero (deja 1.474 € y exige 1.717 € de gasto comercial).</li>
<li>Doblar o quitar el colchón mueve el margen <b>menos de 1 M€</b>: el plan es robusto.</li>
</ul>
</div>
</div>

<h2><span class="num">7</span>Qué vigilar cada mes y qué hacer</h2>
<table class="si">
<tr><th>Señal</th><th style="text-align:left">Qué hacer</th><th style="text-align:left">Por qué</th></tr>
<tr><td>Cuota de Kaido por segmento</td><td class="acc">Si cae 2 meses seguidos, <b>recortar el pedido de t+3</b>.</td><td class="pq">Es la mayor fuente de error (el SUV medio perdió un punto en A012). Un mes puede ser ruido; dos, un cambio real.</td></tr>
<tr><td>Confianza del consumidor</td><td class="acc">Si se recupera, subir el pedido de t+3.</td><td class="pq">Explica casi toda la caída del mercado y se adelanta ≈3 meses.</td></tr>
<tr><td>Faltan coches</td><td class="acc"><b>Transfer si deja dinero</b> (todos menos el pequeño).</td><td class="pq">Perder la venta cuesta más que el 20 % del transfer.</td></tr>
<tr><td>Stock de más de 90 días</td><td class="acc">Parar el pedido de t+3 y <b>no automatricular</b>.</td><td class="pq">Cada coche parado cuesta 60 € al mes y más descuento; automatricular esconde el problema.</td></tr>
<tr><td>Decisión sobre el CO₂</td><td class="acc">Si se confirma, <b>recortar pequeño, compacto y SUV pequeño</b>.</td><td class="pq">Con la prima pierden dinero por coche; SUV medio y compacto siguen ganando. Vale ≈{meur(co2_c, 0)} M€.</td></tr>
</table>

<h2><span class="num">8</span>Recomendaciones al comité</h2>
<div class="reco"><ol>
<li><b>Fabricar para un año de caída, no al ritmo de A012:</b> {es(total_plan)} coches y ≈1 mes de stock en la red (§2, §3).</li>
<li><b>Empujar los tres SUV, reducir el compacto y frenar el pequeño:</b> los SUV dan el {es(100 * suv)} % del margen y siguen ganando con CO₂ (§5).</li>
<li><b>Revisar el plan cada mes</b> con las señales del §7: solo se puede cambiar lo que llega dentro de 3 meses.</li>
<li><b>Reducir las automatriculaciones</b> (≈11 % de las ventas): no son clientes y generan stock viejo con descuento.</li>
<li><b>Pedir a la matriz una decisión sobre la prima de CO₂ cuanto antes</b> y tener el plan B listo: vale ≈{meur(co2_c, 0)} M€ (§6).</li>
</ol></div>


</body></html>
"""

salida = RAIZ / "informe" / "resumen_ejecutivo.html"
salida.write_text(html, encoding="utf-8")
print("ok", salida)
