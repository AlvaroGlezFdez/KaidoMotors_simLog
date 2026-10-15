"""Estilo común de los gráficos (notebooks, informe y presentación).

Cada segmento tiene siempre el mismo color, en todos los gráficos.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS = RAIZ / "salidas" / "figuras"

# Paleta categórica validada (orden fijo, nunca se recicla)
COLORES = {
    "MSUV": "#2a78d6",
    "SSUV": "#eb6834",
    "C": "#1baf7a",
    "B": "#eda100",
    "CSUV": "#e87ba4",
    "MD": "#008300",
    "S": "#6250d6",
}
NOMBRES = {
    "MSUV": "SUV medio", "SSUV": "SUV pequeño", "C": "Compacto", "B": "Pequeño",
    "CSUV": "SUV compacto", "MD": "Berlina media", "S": "Descapotable",
}

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
APAGADO = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
AZUL = "#2a78d6"
ROJO = "#e34948"
SUCESO = "#eb6834"   # meses con sucesos (adelantos, IVA)
BANDA = "#e9e8e2"    # sombreado de periodos (planes de ayuda, caídas)
ALERTA = "#d03b3b"   # umbrales


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 10,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_2,
        "axes.edgecolor": EJE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": REJILLA,
        "grid.linewidth": 0.6,
        "xtick.color": APAGADO,
        "ytick.color": APAGADO,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def es(x, dec=0):
    """Formato español: 1.234,5"""
    if x != x:  # NaN
        return "—"
    s = f"{x + 0.0:,.{dec}f}"  # + 0.0 evita el "-0"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def eje_miles(ax, eje="y"):
    f = FuncFormatter(lambda v, _: es(v))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(f)


def anio_caso(y):
    """2011 -> 'A011'"""
    return f"A{int(y) - 2000:03d}"


def eje_fechas(ax, cada=1):
    """Eje x de fechas ficticias rotulado con los años del caso (A001, A002...)."""
    ax.xaxis.set_major_locator(mdates.YearLocator(cada))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: anio_caso(mdates.num2date(v).year)))


def sombrear(ax, marca, color=BANDA, label=None):
    """Sombrea los tramos en los que la serie binaria 'marca' vale 1."""
    marca = marca.fillna(0).astype(int)
    inicio = None
    fechas = list(marca.index)
    for i, (f, v) in enumerate(marca.items()):
        if v and inicio is None:
            inicio = f
        fin_tramo = v and (i == len(fechas) - 1 or not marca.iloc[i + 1])
        if fin_tramo:
            fin = fechas[i] + (fechas[1] - fechas[0])
            ax.axvspan(inicio, fin, color=color, lw=0, zorder=0, label=label)
            label = None
            inicio = None


def guardar(fig, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / f"{nombre}.png")
