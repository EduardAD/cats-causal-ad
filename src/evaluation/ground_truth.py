# src/evaluation/ground_truth.py

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def load_metadata(meta_path: str) -> pd.DataFrame:
    """
    Carga el metadata.csv del dataset CATS.

    Se espera un CSV con las columnas:
      - 'start_time'
      - 'end_time'
      - 'root_cause'
      - 'affected'
      - 'category'
    """
    df_meta = pd.read_csv(meta_path, parse_dates=['start_time', 'end_time'])
    required = {'start_time', 'end_time', 'root_cause', 'affected', 'category'}
    missing = required - set(df_meta.columns)
    if missing:
        raise ValueError(f"Metadata inválido: faltan columnas {missing}")
    return df_meta

def evaluate_root_cause_detection(G, df_meta):
    """
    Compara el grafo G con la ground truth de cada anomalía.

    Para cada fila de df_meta:
      - Toma 'root_cause' (rc).
      - y_true siempre es 1 (sabemos que hubo anomalía).
      - y_pred = 1 si existe al menos una arista saliente rc → x en G, 0 si no.

    Retorna un dict con precision, recall y f1.
    """
    y_true = [1] * len(df_meta)
    y_pred = []

    for rc in df_meta['root_cause']:
        # Si el nodo rc no está en G, consideramos predicción 0
        if rc not in G:
            y_pred.append(0)
        else:
            # Si tiene al menos una arista saliente
            y_pred.append(1 if G.out_degree(rc) > 0 else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    return {'precision': precision, 'recall': recall, 'f1': f1}
