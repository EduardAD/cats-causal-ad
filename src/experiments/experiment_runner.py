# Experiment Runner for Automated Parameter Sweeps

import os
import sys

# Determina la carpeta actual de ejecución y añade 'src' al path
try:
    current_dir = os.path.dirname(__file__)        # src/experiments cuando se ejecute como script
except NameError:
    current_dir = os.getcwd()                      # fallback en entornos como notebooks
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, src_dir)

import pandas as pd
from src.data.load_data import load_nominal_data
from src.causal.pcmci_inference import run_pcmci_inference
from src.evaluation.metrics import extract_causal_graph, calculate_graph_metrics
from src.evaluation.ground_truth import load_metadata, evaluate_root_cause_detection

def run_experiments(data_file: str,
                    meta_file: str,
                    sample_fracs: list,
                    maxlags: list,
                    alphas: list,
                    random_state: int = 42):
    """
    Ejecuta experimentos de PCMCI para distintas combinaciones de parámetros,
    guarda los resultados en un DataFrame y los exporta a CSV.
    """
    experiments_dir = os.path.join(src_dir, 'experiments')
    os.makedirs(experiments_dir, exist_ok=True)
    df_meta = load_metadata(meta_file)

    records = []
    for sample_frac in sample_fracs:
        # Carga y muestreo
        df = load_nominal_data(data_file, nrows=None,
                               sample_frac=sample_frac,
                               random_state=random_state)
        for maxlag in maxlags:
            for alpha in alphas:
                # Inferencia causal
                pcmci, results_pcmci = run_pcmci_inference(df,
                                                           maxlag=maxlag,
                                                           alpha=alpha)
                # Construcción del grafo
                G = extract_causal_graph(pcmci, results_pcmci,
                                         alpha_level=alpha)
                # Métricas de grafo
                graph_metrics = calculate_graph_metrics(G)
                # Evaluación ground truth (root cause detection)
                gt_scores = evaluate_root_cause_detection(G, df_meta)
                # Registro de resultados
                record = {
                    'sample_frac': sample_frac,
                    'maxlag': maxlag,
                    'alpha': alpha,
                    'num_edges': graph_metrics['num_edges'],
                    'avg_degree': graph_metrics['avg_degree'],
                    'precision': gt_scores['precision'],
                    'recall': gt_scores['recall'],
                    'f1': gt_scores['f1']
                }
                records.append(record)
                print(f"Completed: sf={sample_frac}, lag={maxlag}, α={alpha}")

    # Exportar resultados a CSV
    df_results = pd.DataFrame(records)
    results_path = os.path.join(experiments_dir, 'results.csv')
    df_results.to_csv(results_path, index=False)
    print(f"\nAll experiments completed. Results saved to {results_path}")

if __name__ == "__main__":
    # Rutas a los datos (ajusta a tu estructura)
    DATA_FILE = os.path.join(src_dir, "data", "data.csv")
    META_FILE = os.path.join(src_dir, "data", "metadata.csv")

    # Definición del grid de parámetros
    SAMPLE_FRACS = [0.2, 0.3, 0.4]
    MAXLAGS = [3, 4, 5]
    ALPHAS = [0.05, 0.1]

    run_experiments(DATA_FILE, META_FILE, SAMPLE_FRACS, MAXLAGS, ALPHAS)
