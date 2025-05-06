# descargar_modelo.py

from sentence_transformers import SentenceTransformer

# Nombre del modelo preentrenado de Hugging Face
nombre_modelo = 'distiluse-base-multilingual-cased-v2'

# Cargar el modelo desde internet
print(f"Cargando modelo: {nombre_modelo}")
modelo = SentenceTransformer(nombre_modelo)

# Guardar el modelo localmente en la carpeta ./model_cache
modelo.save('./model_cache')
print("✅ Modelo descargado y guardado en ./model_cache")
