# RAG Pipeline — Lab 2

Retrieval-Augmented Generation over a 10-passage knowledge base.

## Stack

| Layer | Tool |
|-------|------|
| Embeddings | Gemini `text-embedding-004` |
| Vector store | Chroma (in-memory, cosine similarity) |
| Generation | Gemini `gemini-2.0-flash` |

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key"   # never commit this
python rag_pipeline.py
```

## Design Decisions

### Why Chroma with manual embeddings?
Chroma's default embedding function downloads a local ONNX model; using
Gemini embeddings directly avoids that dependency and keeps the embedding
model consistent with the generation model.

### Why chunking isn't needed here — but would be for real docs
Each `knowledge_base.json` entry is already a single focused passage (1–3
sentences, well under the 2 048-token embedding limit).  For full documents
(e.g. a 30-page employee handbook) we'd split into ~300-token windows with
~50-token overlap so:
- no chunk exceeds the embedding model's context window,
- semantically related sentences stay together, and
- facts that span a page break aren't silently truncated.

### Grounding the prompt
The system prompt instructs the model to:
1. answer **only** from the provided context passages,
2. cite the source filename after every fact, and
3. reply *"I don't know — that information isn't in the knowledge base."*
   when the context is silent on the question.

Low temperature (0.2) further reduces hallucination.

## Stretch: top_k=1 vs top_k=3

For a narrow, direct question ("How long do I have to get a full refund?")
top_k=1 is sufficient because a single highly-similar passage contains the
complete answer.

For broader questions the answer may span multiple passages, so top_k=1
would miss important context.  The trade-off: too little context → incomplete
answers; too much context → noisy prompt that dilutes the signal and may push
the model to fabricate connections between unrelated passages.
