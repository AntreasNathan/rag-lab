import requests
from model import search

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, say "Not in documents."

Context:
{context}

Question: {question}
Answer:"""

def generate(question, retrieved):
    context = "\n\n---\n\n".join(
        f"[{c['id']}]\n{c['text']}" for c, _ in retrieved)
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b",
        "prompt": PROMPT.format(context=context, question=question),
        "stream": False,
    }, timeout=120)
    return r.json()["response"]

def answer(question, chunks, emb, k=5):
    hits = search(question, chunks, emb, k=k)
    return generate(question, hits), hits