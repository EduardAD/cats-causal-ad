# src/main.py

import os
from data.load_data import load_nominal_data
from causal.pcmci_inference import run_pcmci_inference
from evaluation.metrics import extract_causal_graph, calculate_graph_metrics
from evaluation.visualization import plot_causal_graph


def main():
    # Ruta del archivo de datos (ajusta la ruta según la ubicación del CSV del dataset CATS)
    data_file = os.path.join("data", "data.csv")

    # Paso 1: Cargar datos nominales
    print("Cargando datos nominales...")
    try:
        df = load_nominal_data(data_file, nrows=1000000)
        print(f"Datos cargados: {df.shape[0]} observaciones, {df.shape[1]} variables.")
    except Exception as e:
        print("Error al cargar los datos. Finalizando el programa.")
        return

    # Paso 2: Ejecutar inferencia causal con PCMCI
    print("Ejecutando inferencia causal con PCMCI...")
    maxlag = 5  # número máximo de retardos a considerar
    alpha = 0.05  # nivel de significación
    pcmci, results = run_pcmci_inference(df, maxlag=maxlag, alpha=alpha)
    print("Inferencia causal completada.")

    # Paso 3: Extraer grafo causal a partir de los resultados
    print("Extrayendo el grafo causal...")
    G = extract_causal_graph(pcmci, results, alpha_level=alpha)
    metrics = calculate_graph_metrics(G)
    print("Métricas del grafo causal:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Paso 4: Visualizar el grafo causal
    print("Mostrando el grafo causal...")
    plot_causal_graph(G, title="Grafo Causal PCMCI")


if __name__ == "__main__":
    main()
