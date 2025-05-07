from sentence_transformers import SentenceTransformer

def cargar_modelo():
    return SentenceTransformer('distiluse-base-multilingual-cased-v2')  # Se descarga automáticamente y se cachea

modelo_nlp = cargar_modelo()
