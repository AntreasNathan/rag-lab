from chunks_gen import load_docs, build_chunks
from model import embed_chunks
from retrieval import answer
from model import search

if __name__ == "__main__":
    docs = load_docs()
    chunks = build_chunks(docs)
    emb = embed_chunks(chunks)

    print(f"{len(chunks)} chunks, embeddings shape {emb.shape}")

    for q in ["What is the vanishing gradient problem?",
              "How does multi-head attention work?"]:
        print(f"\n=== {q}")
        for c, s in search(q, chunks, emb):
            print(f"  {s:.3f}  {c['id']}  {c['text'][:100].strip()}")
        q = "What is the vanishing gradient problem?"


    text, hits = answer(q, chunks, emb)
    print(f"\n=== {q}\n{text}")
    print("\nsources:", [c["id"] for c, _ in hits])