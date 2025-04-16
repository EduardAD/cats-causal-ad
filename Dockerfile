# Dockerfile
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar los archivos del proyecto
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/

# Exponer puerto (en caso de que se añada un dashboard o servidor de visualización)
EXPOSE 8888

# Ejecutar el script principal al iniciar el contenedor
CMD ["python", "src/main.py"]
