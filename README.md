# Kaido Motors España: demanda, stock y beneficio

Caso de uso de la asignatura **Simulación de Sistemas Logísticos** (Universidad Alfonso X el Sabio).
Autores del caso: Jacinto Velasco y Javier Goikoetxea. Profesor: Javier Goikoetxea.

Este repositorio contiene el enunciado y los datos del caso. Es un repositorio de **solo lectura**: clonadlo o descargadlo y trabajad en vuestro propio entorno.

> Caso basado en uno real. Los datos y los nombres se han modificado y camuflado para su uso académico. **Uso exclusivamente docente: no los difundáis fuera del curso.**

## Contenido

```
sim-kaido-motors/
├── README.md
├── requirements.txt
├── .gitattributes
├── .gitignore
├── docs/
│   └── kaido-motors-case-study.pdf    # Enunciado del caso
├── examples/
│   ├── annex4_demand_baseline.ipynb   # Ejemplo del Anexo 4 en notebook (con gráficos)
│   └── annex4_demand_baseline.py      # Mismo ejemplo en script (sin gráficos)
└── data/
    ├── ventas_mensual.csv             # Ventas de Kaido por mes, segmento y canal (A003-A012)
    ├── stock_mensual.csv              # Stock de la red por mes y segmento (A009-A012)
    ├── mercado_mensual.csv            # Mercado por mes, segmento y canal (A001-A012)
    ├── segmentos.csv                  # Ficha de cada uno de los siete segmentos
    ├── codigos.csv                    # Códigos de producto y su agrupación por segmento
    ├── macro_mensual.csv              # Variables económicas, sucesos y calendario (A001-A012)
    └── calendario_A013.csv            # Días laborables y Semana Santa de A013
```

La descripción de cada variable está en el **Anexo 5** del enunciado.

## Puesta en marcha

Necesitáis Python 3.10 o superior.

```bash
git clone <URL-del-repositorio>
cd sim-kaido-motors

python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ejecutad siempre el código **desde la raíz del repositorio**, para que la ruta `data/` se resuelva bien.

## Cómo leer los CSV

Los ficheros están separados por comas. Al leerlos hay que indicar que **solo las celdas vacías son datos que faltan**. Si no, pandas convierte el código de modelo `NA` en un valor nulo (el Anexo 4 del enunciado lo explica):

```python
import pandas as pd

def leer(fichero):
    return pd.read_csv("data/" + fichero, keep_default_na=False, na_values=[""])

mercado = leer("mercado_mensual.csv")
macro = leer("macro_mensual.csv")
```

## Código de ejemplo

El Anexo 4 del enunciado está disponible en dos formatos con el mismo código y las mismas secciones:

- **`examples/annex4_demand_baseline.ipynb`**: notebook **con visualización** (serie, descomposición y comparación con A012).
- **`examples/annex4_demand_baseline.py`**: script **sin visualización**.

Prepara el mercado del SUV medio, lo separa en tendencia, estacionalidad y resto, y ajusta un modelo SARIMAX con dos variables económicas y el calendario. Entrena hasta A011 y compara la previsión con lo que pasó en A012. Es un punto de partida, no la solución del caso.

## Los años del caso

Los años reales se han sustituido por códigos: **A001** es el primer año con datos y **A012** el último cerrado. **A013** es el año que hay que planificar y no tiene datos, salvo el calendario. Para usar librerías de series temporales, el Anexo 4 propone tratar A001 como el año ficticio 2001, A002 como 2002, y así sucesivamente.

Los días laborables y la Semana Santa **no se calculan a partir de la fecha ficticia**: se toman de `macro_mensual.csv` y, para A013, de `calendario_A013.csv`.

## Aviso importante sobre Excel

**No abráis ni guardéis los CSV con Excel.** Puede cambiar el separador (coma por punto y coma), los decimales y el código de modelo `NA` (que convierte en celda vacía). Si queréis echarles un vistazo, usad un editor de texto o pandas.

## Nota de uso

Material de uso docente, exclusivo para el alumnado del curso. No se permite su publicación ni su redistribución sin autorización de los autores.
