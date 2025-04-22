# src/main.py

import os
from data.load_data import load_nominal_data
from causal.pcmci_inference import run_pcmci_inference
from evaluation.metrics import (
    extract_causal_graph,
    calculate_graph_metrics,
    node_degree_stats,
    top_k_hubs
)
from evaluation.visualization import plot_causal_graph
from evaluation.ground_truth import load_metadata, evaluate_root_cause_detection
import matplotlib.pyplot as plt

def main():
    # Rutas de datos
    data_file = os.path.join("..","data", "data.csv")
    meta_file = os.path.join("..","data", "metadata.csv")

    # 1) Carga y preprocesado
    df = load_nominal_data(data_file, nrows=None, sample_frac=0.4, random_state=42)

    # 2) Inferencia causal
    print("Ejecutando inferencia causal con PCMCI...")
    pcmci, results = run_pcmci_inference(df, maxlag=3, alpha=0.05)
    # 3) Extraer grafo causal y calcular métricas
    print("Extrayendo el grafo causal...")
    G = extract_causal_graph(pcmci, results, alpha_level=0.05)
    metrics = calculate_graph_metrics(G)

    # 4) Métricas generales
    print("=== Métricas generales del grafo causal ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 5) Grados y hubs
    deg_stats = node_degree_stats(G)
    hubs = top_k_hubs(G, k=5)
    print("\nTop 5 hubs (mayor out-degree):", hubs['top_hubs'])
    print("Top 5 receptores (mayor in-degree):", hubs['top_receivers'])

    # 6) Carga de ground truth y evaluación de root causes

    print("\nCargando metadata y evaluando Root Cause Channels...")
    df_meta = load_metadata(meta_file)  # ya no falla porque mira 'root_cause'
    gt_scores = evaluate_root_cause_detection(G, df_meta)
    print("=== Evaluación sobre Root Cause Channels ===")
    for metric, score in gt_scores.items():
        print(f"  {metric}: {score:.3f}")

    # 7) Visualizaciones
    print("\nMostrando grafo causal…")
    plot_causal_graph(G, title="Grafo Causal PCMCI")

    # Histograma de grados
    plt.figure(figsize=(8, 4))
    plt.hist(list(deg_stats['in_degrees'].values()), bins=10, alpha=0.7, label='In-degree')
    plt.hist(list(deg_stats['out_degrees'].values()), bins=10, alpha=0.7, label='Out-degree')
    plt.title("Distribución de grados (in vs out)")
    plt.xlabel("Grado")
    plt.ylabel("Número de nodos")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
