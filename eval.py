from chunks_gen import load_docs, build_chunks
from model import embed_chunks
from retrieval import answer, generate
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
    print("\nAnswers are in documents\n")
    for q in questions:
        if q['answer_span'] is None:
            continue
        print(f" Question: {q['question']}")
        found = False
        rank = 0
        chunks_and_scores = search(q['question'], chunks, emb)
        for chunk, score in chunks_and_scores:
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
        llm_answer = generate(q["question"], chunks_and_scores)
        print(f"LLM Answer: {llm_answer}")

    print("Answers NOT in documents\n")
    num_of_no_aswers = 0
    refusals = 0
    for q in questions:
        if q['answer_span'] is not None:
            continue
        num_of_no_aswers += 1
        print(f" Question: {q['question']}")
        chunks_and_scores = search(q['question'], chunks, emb)
        print(f"No Answer in Docs - Score of 1st vector: {chunks_and_scores[0][1]}")
        llm_answer = generate(q["question"], chunks_and_scores)
        print(f"LLM Answer: {llm_answer}")
        if "not in documents" in llm_answer.lower():
            refusals += 1

    recall1 = 0
    recall5 = 0
    mrr = 0
    for r in ranks:
        if r == 1:
            recall1 += 1
        if r is not None:
            recall5 += 1
            mrr += 1/r

    recall1 /= len(ranks)
    recall5 /= len(ranks)
    mrr /= len(ranks)

    abstention = refusals / num_of_no_aswers

    print(f"Recall@1: {recall1:.2f} - Recall@5: {recall5:.2f} - MRR: {mrr:.2f}")
    print(f"Abstention: {abstention:.2f} - Not in docs Questions: {num_of_no_aswers:.2f} - Refusals: {refusals:.2f}")


    # text, hits = answer(q, chunks, emb)
    # print(f"\n=== {q}\n{text}")
    # print("\nsources:", [c["id"] for c, _ in hits])