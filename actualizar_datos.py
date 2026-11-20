"""
Script de actualización de datos bursátiles de Colombia (BVC) mediante Yahoo Finance.
Proyecto de aula: BOHADA & SALAZAR Asesores de Inversión — UNIPAZ (2026).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf

# Lista oficial de acciones colombianas verificadas en Yahoo Finance
ACCIONES_CONFIG = [
    {
        "ticker": "ECOPETROL.CL",
        "nombre": "Ecopetrol",
        "nemotecnico": "ECOPETROL",
        "sector": "Petróleo y gas",
    },
    {
        "ticker": "PFCIBEST.CL",
        "nombre": "Grupo Cibest, preferencial",
        "nemotecnico": "PFCIBEST",
        "sector": "Financiero",
    },
    {
        "ticker": "CIBEST.CL",
        "nombre": "Grupo Cibest, ordinaria",
        "nemotecnico": "CIBEST",
        "sector": "Financiero",
    },
    {
        "ticker": "ISA.CL",
        "nombre": "ISA",
        "nemotecnico": "ISA",
        "sector": "Energía e infraestructura",
    },
    {
        "ticker": "EXITO.CL",
        "nombre": "Grupo Éxito",
        "nemotecnico": "EXITO",
        "sector": "Comercio",
    },
    {
        "ticker": "PFAVAL.CL",
        "nombre": "Grupo Aval, preferencial",
        "nemotecnico": "PFAVAL",
        "sector": "Financiero",
    },
    {
        "ticker": "NUTRESA.CL",
        "nombre": "Grupo Nutresa",
        "nemotecnico": "NUTRESA",
        "sector": "Alimentos procesados",
    },
    {
        "ticker": "GRUPOARGOS.CL",
        "nombre": "Grupo Argos",
        "nemotecnico": "GRUPOARGOS",
        "sector": "Infraestructura",
    },
]

ARCHIVO_SALIDA = "datos.json"


def obtener_trm():
    """Descarga la TRM USD/COP más reciente desde Yahoo Finance."""
    try:
        tk = yf.Ticker("USDCOP=X")
        hist = tk.history(period="10d", auto_adjust=False)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        cierres = hist["Close"].dropna()
        if cierres.empty:
            return None
        ultimo_valor = round(float(cierres.iloc[-1]), 2)
        ultima_fecha = cierres.index[-1].strftime("%Y-%m-%d")
        return {"valor": ultimo_valor, "fecha": ultima_fecha}
    except Exception as e:
        print(f"[AVISO] No se pudo obtener la TRM (USDCOP=X): {e}")
        return None


def procesar_accion(item):
    """Descarga 3 años de precios diarios y calcula todos los indicadores financieros."""
    ticker = item["ticker"]
    tk = yf.Ticker(ticker)
    hist = tk.history(period="3y", auto_adjust=False)

    if hist is None or hist.empty:
        raise ValueError("Sin datos históricos devueltos por Yahoo Finance.")

    # Asegurar que el índice sea DatetimeIndex sin zona horaria para operar fechas limpiamente
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist = hist.sort_index()

    if "Close" not in hist.columns or "Adj Close" not in hist.columns:
        raise ValueError("Faltan columnas Close o Adj Close en la respuesta.")

    cierres = hist["Close"].dropna()
    adj_cierres = hist["Adj Close"].dropna()

    if len(cierres) < 2 or len(adj_cierres) < 2:
        raise ValueError("Insuficientes registros de precios para calcular variaciones.")

    # 1. Precio actual y fecha del último cierre
    precio = round(float(cierres.iloc[-1]), 2)
    fecha_ultima = cierres.index[-1]
    fecha_str = fecha_ultima.strftime("%Y-%m-%d")

    # 2. Variación del día (%)
    variacion_dia_pct = round(((float(cierres.iloc[-1]) / float(cierres.iloc[-2])) - 1.0) * 100.0, 2)

    # 3. Rentabilidad 1 año (%) usando Adj Close de hace ~365 días
    fecha_hace_1a = fecha_ultima - pd.Timedelta(days=365)
    idx_1a = adj_cierres.index.get_indexer([fecha_hace_1a], method="nearest")[0]
    precio_adj_1a = float(adj_cierres.iloc[idx_1a])
    precio_adj_actual = float(adj_cierres.iloc[-1])
    rent_1a_pct = round(((precio_adj_actual / precio_adj_1a) - 1.0) * 100.0, 2)

    # 4. Rentabilidad anual compuesta 3 años (CAGR %) con fechas reales
    fecha_primera = adj_cierres.index[0]
    precio_adj_primero = float(adj_cierres.iloc[0])
    dias_totales = (fecha_ultima - fecha_primera).days
    anios_reales = max(dias_totales / 365.25, 0.1)
    rent_anual_3a_pct = round(
        ((math.pow(precio_adj_actual / precio_adj_primero, 1.0 / anios_reales)) - 1.0) * 100.0,
        2,
    )

    # 5. Volatilidad anual (%) con rendimientos logarítmicos diarios del último año
    adj_ultimo_anio = adj_cierres[adj_cierres.index >= fecha_hace_1a]
    if len(adj_ultimo_anio) >= 2:
        retornos_log = np.log(adj_ultimo_anio / adj_ultimo_anio.shift(1)).dropna()
        volatilidad_anual_pct = round(float(retornos_log.std(ddof=1) * math.sqrt(252)) * 100.0, 2)
    else:
        volatilidad_anual_pct = 0.0

    # 6. Caída máxima en 1 año (Maximum Drawdown negativo %)
    if len(adj_ultimo_anio) >= 2:
        maximo_acumulado = adj_ultimo_anio.cummax()
        drawdowns = (adj_ultimo_anio / maximo_acumulado) - 1.0
        caida_maxima_1a_pct = round(float(drawdowns.min()) * 100.0, 2)
    else:
        caida_maxima_1a_pct = 0.0

    # 7. Rendimiento por dividendo (%) de los últimos 365 días
    if "Dividends" in hist.columns:
        divs_1a = hist.loc[hist.index >= fecha_hace_1a, "Dividends"].fillna(0.0).sum()
        dividendo_rend_pct = round((float(divs_1a) / precio) * 100.0, 2) if precio > 0 else 0.0
    else:
        dividendo_rend_pct = 0.0

    # 8. Histórico mensual: último Close de cada mes de los 3 años
    cierres_mensuales = cierres.resample("ME").last().dropna()
    historico = [
        {"mes": idx.strftime("%Y-%m"), "cierre": round(float(val), 2)}
        for idx, val in cierres_mensuales.items()
    ]

    return {
        "ticker": ticker,
        "nombre": item["nombre"],
        "nemotecnico": item["nemotecnico"],
        "sector": item["sector"],
        "moneda": "COP",
        "precio": precio,
        "fecha": fecha_str,
        "variacion_dia_pct": variacion_dia_pct,
        "rent_1a_pct": rent_1a_pct,
        "rent_anual_3a_pct": rent_anual_3a_pct,
        "volatilidad_anual_pct": volatilidad_anual_pct,
        "caida_maxima_1a_pct": caida_maxima_1a_pct,
        "dividendo_rend_pct": dividendo_rend_pct,
        "historico": historico,
    }


def main():
    print("Iniciando descarga de datos bursátiles de Colombia...")
    trm = obtener_trm()
    acciones_actualizadas = []
    errores = []

    for item in ACCIONES_CONFIG:
        ticker = item["ticker"]
        try:
            datos_acc = procesar_accion(item)
            acciones_actualizadas.append(datos_acc)
            print(f"[OK] {ticker} -> Precio: {datos_acc['precio']} COP ({datos_acc['fecha']})")
        except Exception as e:
            motivo = str(e)
            errores.append({"ticker": ticker, "motivo": motivo})
            print(f"[ERROR] {ticker} -> {motivo}")

    print("\n--- RESUMEN DE ACTUALIZACIÓN ---")
    print(f"Acciones actualizadas con éxito: {len(acciones_actualizadas)} de {len(ACCIONES_CONFIG)}")
    if errores:
        print("Acciones con error:")
        for err in errores:
            print(f"  - {err['ticker']}: {err['motivo']}")

    # Si TODAS las acciones fallaron, no sobrescribir datos.json existente
    if len(acciones_actualizadas) == 0:
        print("[AVISO CRÍTICO] Todas las descargas fallaron. Se conserva el archivo datos.json anterior.")
        sys.exit(0)

    salida = {
        "actualizado": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fuente": "Yahoo Finance (yfinance); rentabilidades con precio ajustado por dividendos",
        "trm": trm,
        "acciones": acciones_actualizadas,
        "errores": errores,
    }

    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)

    print(f"Archivo '{ARCHIVO_SALIDA}' guardado correctamente.")
    sys.exit(0)


if __name__ == "__main__":
    main()