# src/causal/pcmci_inference.py

import numpy as np
import pandas as pd
from tigramite.data_processing import DataFrame as TigDataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

def run_pcmci_inference(df: pd.DataFrame, maxlag: int = 5, alpha: float = 0.05):
    """
    Ejecuta la inferencia causal usando PCMCI sobre el DataFrame dado.

    Parámetros:
      - df: DataFrame de pandas con las variables (columnas) y observaciones (filas).
      - maxlag: número máximo de retardos (lags) a considerar en la inferencia causal.
      - alpha: nivel de significación para los tests de independencia.

    Retorna:
      - pcmci: objeto PCMCI luego de realizar la inferencia.
      - results: diccionario con las conexiones causales detectadas.
    """
    # Convertir el DataFrame a la estructura requerida por Tigramite
    data = df.values
    var_names = list(df.columns)
    dataframe = TigDataFrame(data, var_names=var_names)

    # Configurar el test de independencia a usar (ParCorr)
    parcorr = ParCorr(significance='analytic')

    # Crear el objeto PCMCI
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr)

    # Ejecutar PCMCI
    results = pcmci.run_pcmci(tau_max=maxlag, pc_alpha=alpha)

    return pcmci, results
