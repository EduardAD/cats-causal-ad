# src/data/load_data.py

import pandas as pd

def load_nominal_data(file_path: str, nrows: int = 1000000) -> pd.DataFrame:
    """
    Carga y preprocesa los datos nominales del dataset CATS.

    Parámetros:
      - file_path: ruta al archivo CSV con los datos.
      - nrows: número de filas a leer (por defecto 1 millón para datos nominales).

    Retorna:
      - df: DataFrame con los datos cargados.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        # Aquí se pueden agregar pasos de preprocesamiento,
        # como manejo de fecha-hora, normalización, etc.
        # Por ejemplo:
        # df['timestamp'] = pd.to_datetime(df['timestamp'])
        # df = df.set_index('timestamp')
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise
