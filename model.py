import numpy as np
import os
from sentence_transformers import SentenceTransformer

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model

def embed_chunks(chunks, cache="embeddings.npy"):
    if os.path.exists(cache):
        return np.load(cache)
    texts = [c["text"] for c in chunks]
    emb = get_model().encode(texts, normalize_embeddings=True,
                             batch_size=32, show_progress_bar=True)
    np.save(cache, emb)
    return emb

def search(query, chunks, emb, k=5):
    q = get_model().encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]