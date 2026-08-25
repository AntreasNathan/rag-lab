import os, glob

def load_docs(folder="docs"):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        docs.append({"source": os.path.basename(path), "text":text})
    return docs

def chunk_text(text, size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def build_chunks(docs):
    out = []
    for d in docs:
        for i, c in enumerate(chunk_text(d["text"])):
            out.append({
                "id" : f"{d['source']}::{i}",
                "source":d["source"],
                "text": c,
            })
    return out