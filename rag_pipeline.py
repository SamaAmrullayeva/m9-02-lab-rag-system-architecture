"""
RAG Pipeline — Lab 2
Retrieval-Augmented Generation over a small knowledge base using
Chroma (vector store) + Gemini (embeddings + generation).

No API key is stored in this file. Set GOOGLE_API_KEY in the environment:
    export GOOGLE_API_KEY="your-key"
"""

import json
import os
import sys
import textwrap
import chromadb
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. INDEXING
# ---------------------------------------------------------------------------
# Why chunking matters:
#   Each entry in knowledge_base.json is already a self-contained passage (~1-3
#   sentences), so no further splitting is needed here.  If these were full
#   documents (e.g. a 30-page handbook), we'd need to split them into ~200-400
#   token windows with some overlap so that:
#     a) no single chunk exceeds the embedding model's token limit,
#     b) semantically related sentences are kept together, and
#     c) context that spans a page boundary isn't silently truncated.

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")


def load_knowledge_base(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def embed_texts(texts: list[str], gemini: genai.Client) -> list[list[float]]:
    """Embed a batch of texts using the Gemini embedding model."""
    result = gemini.models.embed_content(
        model="models/text-embedding-004",
        contents=texts,
    )
    return [e.values for e in result.embeddings]


def build_index(kb: list[dict], gemini: genai.Client) -> chromadb.Collection:
    """
    Embed each passage via Gemini and store in a Chroma in-memory collection.
    Metadata (source filename) is stored alongside each document so we can
    surface it in citations later.
    """
    client = chromadb.Client()  # ephemeral / in-memory

    # We supply our own embeddings so Chroma doesn't need a local model.
    collection = client.create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,   # we pass embeddings manually
    )

    texts      = [item["text"]   for item in kb]
    ids        = [item["id"]     for item in kb]
    metadatas  = [{"source": item["source"]} for item in kb]
    embeddings = embed_texts(texts, gemini)

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"[index] Added {collection.count()} passages to the vector store.\n")
    return collection


# ---------------------------------------------------------------------------
# 2. QUERYING — retrieve + assemble prompt
# ---------------------------------------------------------------------------

def retrieve(question: str, collection: chromadb.Collection,
             gemini: genai.Client, top_k: int = 3) -> list[dict]:
    """
    Embed the question and return the top_k most similar passages
    together with their source metadata.
    """
    q_emb = embed_texts([question], gemini)[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    passages = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({"text": doc, "source": meta["source"], "distance": dist})
    return passages


def build_prompt(question: str, passages: list[dict]) -> str:
    """
    Assemble a grounded prompt:
    - instruct the model to answer ONLY from the provided context,
    - include each retrieved passage tagged with its source,
    - ask it to cite sources and say "I don't know" when context is absent.
    """
    context_block = "\n\n".join(
        f"[Source: {p['source']}]\n{p['text']}" for p in passages
    )
    prompt = textwrap.dedent(f"""
        You are a helpful assistant.  Answer the question below using ONLY the
        context passages provided.  For every fact you state, cite the source
        file name in parentheses, e.g. (handbook.md).  If the context does not
        contain enough information to answer the question, respond with exactly:
        "I don't know — that information isn't in the knowledge base."
        Do NOT invent or infer facts beyond what the passages say.

        --- CONTEXT ---
        {context_block}
        --- END CONTEXT ---

        Question: {question}
    """).strip()
    return prompt


# ---------------------------------------------------------------------------
# 3. GENERATION
# ---------------------------------------------------------------------------

def generate_answer(prompt: str, gemini: genai.Client) -> str:
    response = gemini.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=512,
            temperature=0.2,   # low temp → more faithful / less hallucination
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# 4. FULL PIPELINE — retrieve → augment → generate
# ---------------------------------------------------------------------------

def ask(question: str, collection: chromadb.Collection, gemini: genai.Client,
        top_k: int = 3) -> dict:
    passages = retrieve(question, collection, gemini, top_k=top_k)
    prompt   = build_prompt(question, passages)
    answer   = generate_answer(prompt, gemini)
    return {"question": question, "passages": passages, "answer": answer}


# ---------------------------------------------------------------------------
# 5. DISPLAY + PERSISTENCE
# ---------------------------------------------------------------------------

def print_result(result: dict) -> None:
    SEP = "─" * 72
    print(SEP)
    print(f"Q: {result['question']}")
    print()
    print("Retrieved sources:")
    for i, p in enumerate(result["passages"], 1):
        score = 1 - p["distance"]   # cosine similarity from cosine distance
        snippet = p["text"][:90] + ("…" if len(p["text"]) > 90 else "")
        print(f"  {i}. [{p['source']}]  similarity={score:.3f}")
        print(f"     {snippet}")
    print()
    print(f"Answer:\n{result['answer']}")
    print()


def save_results(results: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    for r in results:
        lines.append("=" * 72)
        lines.append(f"Q: {r['question']}")
        lines.append("")
        lines.append("Retrieved sources:")
        for i, p in enumerate(r["passages"], 1):
            score = 1 - p["distance"]
            lines.append(f"  {i}. [{p['source']}]  similarity={score:.3f}")
            lines.append(f"     {p['text']}")
        lines.append("")
        lines.append(f"Answer:\n{r['answer']}")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] Results written to {path}")


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Error: GOOGLE_API_KEY environment variable is not set.")

    gemini = genai.Client(api_key=api_key)

    # --- Index ---
    kb         = load_knowledge_base(KB_PATH)
    collection = build_index(kb, gemini)

    # --- Required questions ---
    questions = [
        "How long do I have to get a full refund?",        # answerable
        "How do I reset my password?",                      # answerable
        "What is the company's stock price today?",         # out-of-scope
    ]

    results = []
    for q in questions:
        r = ask(q, collection, gemini, top_k=3)
        results.append(r)
        print_result(r)

    # --- Save required results ---
    save_results(results, "/mnt/user-data/outputs/results.txt")

    # --- Optional stretch: top-1 vs top-3 ---
    print("\n" + "=" * 72)
    print("STRETCH: top_k=1 vs top_k=3 for the refund question")
    print("=" * 72 + "\n")

    q_stretch = "How long do I have to get a full refund?"
    r_top1 = ask(q_stretch, collection, gemini, top_k=1)
    r_top3 = ask(q_stretch, collection, gemini, top_k=3)

    print("--- top_k=1 ---")
    print_result(r_top1)
    print("--- top_k=3 ---")
    print_result(r_top3)

    print(
        "Trade-off note:\n"
        "  With top_k=1 the model only sees the single most-similar passage and\n"
        "  can still answer a narrow, direct question correctly.  But for broader\n"
        "  questions — e.g. 'What are my options after a purchase?' — it would miss\n"
        "  the subscription-cancellation context that ranks 2nd or 3rd.  Too little\n"
        "  context risks incomplete answers; too much context (large top_k) risks\n"
        "  diluting the signal with loosely-related passages that confuse the model\n"
        "  or bloat the prompt beyond what the LLM can attend to effectively.\n"
    )


if __name__ == "__main__":
    main()
