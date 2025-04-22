# src/data/load_data.py

import pandas as pd


def load_nominal_data(file_path: str, nrows: int = None, sample_frac: float = None, random_state: int = 42) -> pd.DataFrame:
    """
    Carga y preprocesa los datos nominales del dataset CATS.

    Se asume que:
      - La primera columna es 'timestamp' y se elimina.
      - Las dos últimas columnas son 'etiqueta' y 'tipo de error' y se eliminan.

    Parámetros:
      - file_path: ruta al archivo CSV con los datos.
      - nrows: número de filas a leer (por defecto 1 millón de observaciones para datos nominales).

    Retorna:
      - df: DataFrame con las variables útiles para la inferencia causal.
    """
    try:
        df = pd.read_csv(file_path, nrows=nrows)
        # Si sample_frac está definido, muestreamos aleatoriamente esa fracción

        if sample_frac is not None and 0 < sample_frac < 1:
                   df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)        # Imprimir los nombres de las columnas antes de preprocesar para verificar el orden
        print("Columnas originales:", df.columns.tolist())

        # Solo proceder si hay por lo menos 3 columnas
        if len(df.columns) >= 3:
            # Suponemos que la primera columna es el timestamp
            # y las dos últimas son etiqueta y tipo de error; por ello las removemos.
            cols_to_drop = [df.columns[0], df.columns[-2], df.columns[-1]]
            df = df.drop(columns=cols_to_drop)
            print("Columnas después del preprocesamiento:", df.columns.tolist())
        else:
            print("Advertencia: Se esperan al menos 3 columnas. No se eliminó ninguna columna.")

        # Aquí se pueden aplicar transformaciones adicionales si es necesario
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise
