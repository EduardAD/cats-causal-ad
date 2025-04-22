# src/evaluation/metrics.py

import numpy as np
import networkx as nx
from tigramite.pcmci import PCMCI


def extract_causal_graph(pcmci: PCMCI, results: dict, alpha_level: float = 0.05):
    """
    Extrae una representación del grafo causal basado en los resultados de PCMCI.

    Parámetros:
      - pcmci: objeto PCMCI usado en la inferencia.
      - results: resultados devueltos por run_pcmci.
      - alpha_level: nivel de significación para filtrar conexiones.

    Retorna:
      - G: un grafo de NetworkX con las conexiones causales detectadas.
    """
    # Los resultados contienen una matriz de valores p para cada par (variable, lag)
    # Extraemos las conexiones que son significativas.
    # En Tigramite, results['p_matrix'] es una matriz de p-valores donde cada entrada
    # corresponde a un par (variable_j, lag) para cada variable_i.

    p_matrix = results['p_matrix']
    var_names = pcmci.dataframe.var_names
    maxlag = p_matrix.shape[2]  # número de lags considerados

    G = nx.DiGraph()
    for var in var_names:
        G.add_node(var)

    # Recorrer cada combinación (i, j, lag)
    for i, cause in enumerate(var_names):
        for j, effect in enumerate(var_names):
            for lag in range(1, maxlag):  # lag 0 no se considera
                if p_matrix[i, j, lag] < alpha_level:
                    # Añadimos conexión: desde 'cause' en tiempo t-lag a 'effect' en tiempo t
                    label = f"lag_{lag}"
                    G.add_edge(cause, effect, lag=lag, p_val=p_matrix[i, j, lag])
    return G


def calculate_graph_metrics(G: nx.DiGraph) -> dict:
    """
    Calcula métricas básicas del grafo causal.

    Parámetros:
      - G: grafo de NetworkX.

    Retorna:
      - métricas: diccionario con métricas (número de aristas, grado medio, etc.).
    """
    num_edges = G.number_of_edges()
    degrees = [deg for node, deg in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0
    metrics = {
        'num_edges': num_edges,
        'avg_degree': avg_degree
    }
    return metrics

def node_degree_stats(G: nx.DiGraph) -> dict:
    """
    Calcula in-degree y out-degree de cada nodo.

    Retorna:
      {
        'in_degrees': dict nodo→in-degree,
        'out_degrees': dict nodo→out-degree
      }
    """
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    return {'in_degrees': in_degrees, 'out_degrees': out_degrees}

def top_k_hubs(G: nx.DiGraph, k: int = 5) -> dict:
    """
    Obtiene los k nodos con mayor out-degree (hubs) y con mayor in-degree (receptores).

    Retorna:
      {
        'top_hubs': [(nodo, out_degree), …],
        'top_receivers': [(nodo, in_degree), …]
      }
    """
    degs = node_degree_stats(G)
    top_hubs = sorted(degs['out_degrees'].items(), key=lambda x: x[1], reverse=True)[:k]
    top_receivers = sorted(degs['in_degrees'].items(), key=lambda x: x[1], reverse=True)[:k]
    return {'top_hubs': top_hubs, 'top_receivers': top_receivers}
