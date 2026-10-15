# Kaido Motors España · Plan de producción A013

Resolución del caso **Kaido Motors España: demanda, stock y beneficio** de la asignatura *Simulación de Sistemas Logísticos* (Universidad Alfonso X el Sabio, curso 2026/27).

Autor: Álvaro González Fernández · 4º de Ingeniería Matemática.

## Objetivo

Proponer cuántos coches fabricar cada mes de A013 en cada uno de los siete segmentos de Kaido. Para ello se construye un modelo de demanda mensual y se simula el año con tres escenarios (pesimista, central y optimista), midiendo ventas, stock, transfers, gasto comercial y margen.

## Resultados principales

| | Pesimista | Central | Optimista |
|---|---|---|---|
| Coches vendidos | 18.878 | 22.613 | 27.678 |
| Coches fabricados (llegadas a la red) | 16.617 | 19.379 | 22.804 |
| Stock medio en la red (meses de venta) | 1,4 | 1,1 | 0,8 |
| Margen de la marca, CVP (M€) | 4,3 | 81,8 | 107,4 |

- **Plan de partida:** 20.254 coches en A013, con producción del compacto parada de marzo a agosto y un pedido alto de SUV pequeño en marzo.
- **Mercado A013:** −14 % frente a A012, por la caída de la confianza del consumidor y la devolución de las compras adelantadas.
- **Riesgo principal:** la prima europea de CO₂, que costaría ≈67 M€ con las ventas del escenario central.

El resumen ejecutivo y la presentación están en `entregables/`.

## Estructura

```
├── run.py                     Pipeline completo: previsión, plan y simulación → salidas/
├── src/
│   ├── datos.py               Carga de los CSV y construcción de la demanda limpia
│   ├── demanda.py             Modelo de mercado (SARIMAX), peso de segmento y cuota
│   ├── simulacion.py          Escenarios, política de pedidos y simulación Monte Carlo
│   └── graficos.py            Estilo común de los gráficos
├── notebooks/
│   ├── 01_exploracion_demanda.ipynb   Demanda real de A012: automatriculaciones, adelantos y roturas de stock
│   ├── 02_modelo_demanda.ipynb        Selección de variables, modelo y validación con A012
│   └── 03_simulacion_A013.ipynb       Escenarios, plan de producción y Tabla 5
├── informe/
│   ├── generar_resumen.py     Genera el resumen ejecutivo (HTML → PDF)
│   └── generar_presentacion.py
├── salidas/                   Tablas (plan, previsión, Tabla 5) y figuras
├── entregables/               Resumen ejecutivo y presentación en PDF
└── requirements.txt
```

## Cómo reproducirlo

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Los datos del caso son material docente de la asignatura y **no se incluyen en el repositorio**. Para ejecutar el código hay que copiar los siete CSV del repositorio del curso en `data/`. Todas las cifras se reproducen con semilla fija (13).

Para regenerar los PDF:

```bash
python informe/generar_resumen.py
python informe/generar_presentacion.py
# imprimir informe/*.html a PDF (por ejemplo, con Chrome: --headless --print-to-pdf)
```

## Metodología

1. **Demanda real.** Ventas menos automatriculaciones. Los meses con compras adelantadas por cambios de normativa y los meses en que la red se quedó sin coches se marcan aparte.
2. **Previsión.** `demanda = mercado total × peso del segmento × cuota de Kaido`.
   - **Mercado total:** SARIMAX sobre el logaritmo, con calendario, planes de ayuda, subidas del IVA y dos variables económicas (confianza del consumidor y paro) con tres meses de retraso.
   - **Peso del segmento:** peso del último año corregido con la tendencia de los dos últimos.
   - **Cuota:** media de los últimos doce meses.
   - **Validación:** entrenando hasta A011 y prediciendo A012.
3. **Plan mensual.** Cada mes se decide la llegada de t+3 con `pedido = máx(0, demanda prevista + stock de seguridad − stock previsto)`. El stock de seguridad de cada segmento sale del ratio crítico del problema del *newsvendor*.
4. **Simulación.** 1.000 años por escenario, con error de demanda (sesgo de nivel y ruido mensual) calibrado en la validación, revisión mensual de la previsión, transfers cuando son rentables y seguimiento de la antigüedad del stock (FIFO).

## Limitaciones

- Cada segmento se trata como un único producto.
- Se supone que los transfers están siempre disponibles.
- Las llegadas en camino de enero y febrero se estiman con el *wholesale* de octubre a diciembre de A012.
- El modelo se ha validado con un solo año.
