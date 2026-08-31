# rag-with-eval

A retrieval-augmented generation pipeline built from scratch, with an eval harness that measures how each design choice affects retrieval quality.

No LangChain, no LlamaIndex, no vector database. The retrieval loop is about 15 lines of numpy. The point of the project was not to assemble a RAG but to measure one, so most of the work went into the evaluation.

Everything runs locally: embeddings on CPU via sentence-transformers, generation via Ollama. No API keys, no external calls.

## Results

| Configuration | Chunks | recall@5 | recall@1 | MRR |
|---|---|---|---|---|
| Fixed-size chunking (1000 char, 150 overlap) | 435 | 0.60 | 0.27 | 0.37 |
| + LaTeX cleaning | 406 | 0.60 | 0.33 | 0.44 |
| + sentence-aware chunking | 474 | **0.80** | **0.53** | **0.65** |

Abstention accuracy: **5/5**. All five questions whose answers are absent from the corpus were correctly refused.

Measured over 20 hand-verified questions (15 answerable, 5 not) against a corpus of 10 Wikipedia articles on machine learning, roughly 735k characters.

## Stack

- **Embeddings:** `BAAI/bge-small-en-v1.5` via sentence-transformers, 384 dimensions, CPU
- **Vector store:** a numpy array. At 474 chunks a dot product scores the whole corpus in one operation, so a vector DB would have added dependencies and no speed.
- **Generation:** `llama3.1:8b` served locally by Ollama
- **Corpus:** 10 Wikipedia ML articles pulled through the MediaWiki API

## Pipeline

```
Wikipedia API
      |
  clean_text()        collapse whitespace, strip {\displaystyle ...} blocks
      |
  chunk_sentences()   split on sentence boundaries, pack to ~1000 chars,
      |               carry one sentence of overlap
  embed_chunks()      bge-small -> (474, 384), normalised, cached to disk
      |
  search()            embed query, dot product against all chunks, top 5
      |
  generate()          stuff the 5 chunks into a prompt, call Ollama
```

## How the evaluation works

The golden set ties each question to an **answer span**, a short verbatim phrase from the source documents, rather than to a chunk ID.

```json
{
  "id": "q2",
  "question": "What three gates make up an LSTM unit?",
  "answer_span": "an input gate, an output gate, and a forget gate",
  "source_doc": "long_short-term_memory.txt"
}
```

A retrieval counts as correct if any of the top-k chunks contains that span as a substring.

This matters because the whole project is about changing the chunking. Chunk IDs shift the moment you alter chunk size, add cleaning, or switch splitting strategy, so an eval keyed on IDs silently compares against the wrong thing after the first ablation. The span lives in the document, not in the configuration, so the same golden set works across every setup tested here.

Every span was verified to appear exactly once in the corpus before being used.

Three metrics:

- **recall@5** — did any of the top 5 chunks contain the answer
- **recall@1** — was it ranked first
- **MRR** — mean of 1/rank, 0 for a miss

The five unanswerable questions are scored separately, on whether the model refuses. Mixing them into recall would just deflate the number by a quarter for no reason.

## Findings

### Cleaning LaTeX improved ranking but found nothing new

Wikipedia's plaintext export mangles rendered maths. Formulas come out as one symbol per line followed by the raw `{\displaystyle ...}` source, so a 1000-character window can be almost entirely symbol soup.

On day one, before any cleaning, the query *"what is the vanishing gradient problem"* returned five chunks of pure LaTeX, all scoring 0.76–0.77. That is a strong-looking cosine score for text with no meaning in it. Low-information chunks sit near the centroid of the embedding space, so they are moderately close to everything and never strongly far from anything. A chunk that actually discusses vanishing gradients points in a specific direction: it wins on the right query and loses everywhere else. The junk plays it safe and wins by default.

Stripping the `{\displaystyle}` blocks left recall@5 unchanged at 0.60 but moved MRR from 0.37 to 0.44. It did not make more answers findable; it stopped junk from outranking the real ones. Four questions moved up (ranks 3→2, 5→2, 2→1, 4→3) and one recovered entirely.

**If I had only tracked recall@5 I would have concluded this change did nothing and reverted it.**

It was not a free win either. One question regressed from rank 5 to a miss, because the maths blocks in the backpropagation article sit directly beside the text naming the loss functions, and the regex took some of that with it.

### Sentence-aware chunking was the real fix

Cutting text every 1000 characters puts boundaries in the middle of sentences. When the sentence containing the answer gets split, no single chunk holds the complete phrase, so it cannot be retrieved intact regardless of how good the embedding model is.

Replacing the fixed-size chunker with one that accumulates whole sentences up to a size limit took recall@5 from 0.60 to 0.80 and MRR from 0.44 to 0.65. Four questions that were previously invisible became findable, and eight questions now rank first, up from four.

### The similarity threshold does not work

Retrieval always returns k chunks. Ask about PageRank and you still get five results with respectable scores, which is the path to a confident, fabricated answer. The obvious defence is a score floor: if the top chunk scores below some threshold, refuse before calling the model.

I measured the two distributions:

- Answerable questions, top-1 score: **0.71 – 0.91**
- Unanswerable questions, top-1 score: **0.66 – 0.78**

They overlap, and not narrowly. The question about denoising diffusion models scored **0.776**, higher than 7 of the 12 successful retrievals. Any threshold that catches it also rejects questions the system currently answers correctly.

So the threshold was dropped. Prompt-level refusal reached 5/5 on its own. The underlying reason is the same anisotropy that made the LaTeX chunks score highly: **cosine similarity is a ranking signal, not an absolute measure of relevance.** A chunk can be the best available match and still be useless.

The diffusion question is a deliberate hard negative. The corpus mentions the phrase "diffusion model" exactly once, in a sentence saying DALL-E is not one. Retrieval surfaces that chunk, the term matches, and the model still refuses.

## Failure analysis

Three questions still miss at k=5. All three share a mechanism.

| Question | Target chunk |
|---|---|
| What is the vanishing gradient problem in RNNs? | `recurrent_neural_network.txt::1` |
| What three gates make up an LSTM unit? | `long_short-term_memory.txt::0` |
| How do soft weights differ from hard weights? | `attention_machine_learning.txt::0` |

All three are early summary chunks, and all three are broad. The LSTM intro covers the definition, the naming analogy, the vanishing gradient motivation, the three gates, and what the cell does. The RNN chunk covers applications, the vanishing gradient problem, LSTM's history, GRUs, and transformers — five topics in 890 characters. Averaged into one 384-dimensional vector, the sentence that answers the question contributes maybe 15% of its direction.

Meanwhile the equations chunk further down is entirely about gates. It points squarely at the query and wins, despite not containing the answer sentence.

The competition is not one correct chunk against 473 random ones. It is one diluted chunk against a dozen focused chunks on the same subject, and the corpus is ten articles on deliberately overlapping topics.

All three failures share exact query terms with their target chunk: "forget gate", "soft weights", "vanishing gradient problem". Keyword matching would rank them highly where dense retrieval buries them, which makes hybrid retrieval the obvious next step.

The generator answered all three correctly anyway, using the retrieved context. So the content was reaching the model even when exact span matching scored it as a miss, meaning true retrieval quality is somewhat higher than 0.80.

## Limitations

- **Sentence splitting is a regex.** `(?<=[.!?])\s+` misfires on "Fig. 2", "et al." and similar. A proper sentence tokenizer would handle these.
- **Span matching is exact substring.** Brittle against curly versus straight quotes and against chunk boundaries. It undercounts.
- **One span per question.** Several facts appear in more than one article. The vanishing gradient explanation is in both the RNN and LSTM pages; retrieval found the LSTM version and was scored as a miss because the golden entry points at the RNN one. Accepting a list of spans would fix this.
- **20 questions is a small sample.** One question is worth 6.7 points of recall, so differences under about 0.10 are not meaningful.
- **The abstention check is a string match** for "not in documents". It would not catch a differently-worded refusal.
- **Single embedding model, single corpus.** These numbers describe this setup, not chunking in general.

## Next steps

1. **Hybrid BM25 + dense with reciprocal rank fusion.** The diagnosis above predicts this recovers the three misses. RRF combines ranks rather than raw scores, which sidesteps the fact that cosine similarity and BM25 scores are not on comparable scales.
2. **Cross-encoder reranking.** Retrieve 20 with the bi-encoder, rescore with `bge-reranker-base`, keep 5. recall@1 at 0.53 against recall@5 at 0.80 is exactly the gap reranking is for.
3. **Chunk size sweep.** 500 / 1000 / 2000 to see the precision-versus-context curve directly.
4. **Multiple accepted spans per question**, to stop scoring correct retrievals as misses.

## Running it

```bash
conda create -n rag python=3.11 -y
conda activate rag
pip install sentence-transformers numpy requests

python get_docs.py     # pulls 10 Wikipedia articles into docs/
python eval.py         # runs the full evaluation
```

Generation needs Ollama running with the model pulled:

```bash
ollama pull llama3.1:8b
```

Delete `embeddings.npy` after any change to chunking. The cache is keyed on nothing and will otherwise be silently stale.

## Files

| File | Purpose |
|---|---|
| `get_docs.py` | Fetches the corpus from the MediaWiki API |
| `chunks_gen.py` | Cleaning, sentence splitting, both chunkers |
| `model.py` | Embedding, caching, dense search |
| `retrieval.py` | Prompt construction and the Ollama call |
| `eval.py` | The harness: recall@1, recall@5, MRR, abstention |
| `find.py` | Corpus grep. Used to locate and verify answer spans |
| `questions.json` | The 20-question golden set |