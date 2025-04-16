# cats-causal-ad

## Requisitos

- Docker (opcional, para correr el contenedor)
- Python 3.9 o superior

## Instalación

1. Clona el repositorio de GitHub:
    ```bash
    git clone https://github.com/tu_usuario/cats_causal_project.git
    cd cats_causal_project
    ```

2. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Ejecución

- **Localmente:**  
  Ejecuta:
    ```bash
    python src/main.py
    ```

- **Con Docker:**  
  Construye la imagen:
    ```bash
    docker build -t cats_causal_project .
    ```
  Ejecuta el contenedor:
    ```bash
    docker run -it --rm cats_causal_project
    ```

## Descripción del flujo

1. **Carga de datos:** Se carga un subconjunto del dataset CATS (el segmento nominal) mediante `src/data/load_data.py`.
2. **Inferencia causal:** Se realiza la inferencia causal con PCMCI en `src/causal/pcmci_inference.py`.
3. **Evaluación:** Se calculan métricas y se visualizan los resultados en `src/evaluation/metrics.py` y `src/evaluation/visualization.py`.
4. **Orquestación:** `src/main.py` integra todas las etapas.

## Contribuciones

Se anima la colaboración a través de pull requests y issues.

## Licencia

[Licencia MIT](LICENSE)
