from chunks_gen import load_docs, build_chunks
from model import embed_chunks
from retrieval import answer
from model import search
import json



if __name__ == "__main__":
    docs = load_docs()
    chunks = build_chunks(docs)
    emb = embed_chunks(chunks)
    

    with open("questions.json", "r" , encoding="utf-8") as file:
        questions = json.load(file)
    

    # print(f"{len(docs)} docs → {len(chunks)} chunks")
    # print("\n--- FIRST CHUNK ---")
    # print(chunks[0]["text"][:400])
    # print("\n--- A MIDDLE CHUNK ---")
    # print(chunks[len(chunks)//2]["text"][:400])

    print(f"{len(chunks)} chunks, embeddings shape {emb.shape}")

    # for q in ["What is the vanishing gradient problem?",
    #           "How does multi-head attention work?"]:
    #     print(f"\n=== {q}")
    #     for c, s in search(q, chunks, emb):
    #         print(f"  {s:.3f}  {c['id']}  {c['text'][:100].strip()}")
    #     q = "What is the vanishing gradient problem?"

    ranks = []

    for q in questions:
        print(f" Question: {q['question']}")
        found = False
        if q['answer_span'] is None:
            continue
        rank = 0
        for chunk, score in search(q['question'], chunks, emb):
            #print(f"Text ----- {chunk['text']}")
            rank += 1
            if q['answer_span'] in chunk['text']:
                print(f"Found! - Position: {rank} - Score: {score}")
                found = True
                ranks.append(rank)
                break
        if not found:
            ranks.append(None)
            print("Not found")

    recall1 = 0
    recall5 = 0
    mmr = 0
    for r in ranks:
        if r == 1:
            recall1 += 1
        if r is not None:
            recall5 += 1
            mmr += 1/r

    recall1 /= len(ranks)
    recall5 /= len(ranks)
    mmr /= len(ranks)

    print(f"Recall@1: {recall1:.2f} - Recall@5: {recall5:.2f} - MMR: {mmr:.2f}")

    # text, hits = answer(q, chunks, emb)
    # print(f"\n=== {q}\n{text}")
    # print("\nsources:", [c["id"] for c, _ in hits])