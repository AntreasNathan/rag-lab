import os, glob, re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\{\\displaystyle.*?\}", " ", text, flags=re.S)
    return text

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)

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


def chunk_sentences(text, size=1000, overlap=150):
    sentences = split_sentences(text)
    chunks = []
    buffer = []
    last_sentence = ""
    length = 0

    for s in sentences:

        if s is not None and len(s) + length > size:
            chunks.append(" ".join(buffer))
            buffer = []
            buffer.append(last_sentence)
            length = len(last_sentence)
        last_sentence = s
        buffer.append(s) 
        length += len(s)

    if len(buffer) > 0:
        chunks.append(" ".join(buffer))

    return chunks

    
def build_chunks(docs):
    out = []
    for d in docs:
        text = clean_text(d["text"])
        # text = split_sentences(text)
        for i, c in enumerate(chunk_sentences(text)):
            out.append({
                "id" : f"{d['source']}::{i}",
                "source":d["source"],
                "text": c,
            })
    return out
