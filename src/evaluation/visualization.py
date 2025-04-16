# src/evaluation/visualization.py

import matplotlib.pyplot as plt
import networkx as nx

def plot_causal_graph(G: nx.DiGraph, title: str = "Grafo Causal PCMCI"):
    """
    Genera y muestra un gráfico del grafo causal.

    Parámetros:
      - G: Grafo de NetworkX.
      - title: Título del gráfico.
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout para distribución de nodos
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis('off')
    plt.show()
