"""
Global Economic Prospects (June 2026) Q&A — Use Case 2: a long, structured report

  Question -> Query understanding (Gemini: region / chapter filter + search rewrite)
           -> Hybrid search inside the filter (dense + BM25, fused with RRF)
           -> Gemini rerank -> Swap children for their parent sections
           -> Numbered passages with heading paths + Conversation History -> Gemini -> Cited answer

Ingestion (text repair, outline from fonts, parents + children) lives in ingest.py.
The chunks and their vectors are cached in .index/ after the first run.

Run:
    uv run python chatbot.py
"""

import os
import re
import json
import logging
from getpass import getpass

# Silence the google-genai SDK's benign "use Chat.send_message instead" advisory
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from rank_bm25 import BM25Okapi
from fastembed import TextEmbedding

from ingest import REGIONS, build_chunks, with_heading


# ---------------------------------------------------------------------------
# 1. API key (never hardcoded)
# ---------------------------------------------------------------------------
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Gemini API key: ")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


def get_text(response):
    """Return plain text of an LLM response, whether `.content` is a string
    or a list of content blocks (e.g. text + an internal thought-signature block)."""
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# ---------------------------------------------------------------------------
# 2. LLM — temperature 0: answers must quote the document, not paraphrase it
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)


# ---------------------------------------------------------------------------
# 3. Embeddings + vector store (built once, then loaded from .index/)
#    Children are what we search; each embeds its heading path in front of its text.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "GEP-Jun-2026.pdf")
INDEX_DIR = os.path.join(BASE_DIR, ".index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.json")
VECTORS_PATH = os.path.join(INDEX_DIR, "vectors.json")


class LocalEmbeddings(Embeddings):
    """BAAI/bge-small-en-v1.5 via fastembed: runs locally on CPU (ONNX), so indexing has
    no API quota. The model (~67 MB) is downloaded once on first use."""

    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.query_embed(text))).tolist()


embeddings = LocalEmbeddings()

if os.path.exists(CHUNKS_PATH) and os.path.exists(VECTORS_PATH):
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        saved = json.load(f)
    children, parents = saved["children"], saved["parents"]
    vector_store = InMemoryVectorStore.load(VECTORS_PATH, embeddings)
else:
    children, parents = build_chunks(PDF_PATH)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(
        [Document(page_content=with_heading(c), metadata={"id": c["id"]}) for c in children],
        ids=[str(c["id"]) for c in children],
    )
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump({"children": children, "parents": parents}, f)
    vector_store.dump(VECTORS_PATH)


# ---------------------------------------------------------------------------
# 4. Query understanding: one Gemini call -> filter fields + a search-friendly rewrite
# ---------------------------------------------------------------------------
QUERY_PROMPT = """You turn a question about the World Bank's Global Economic Prospects report (June 2026)
into search instructions. Every passage in the report is tagged with where it lives:

- region (Chapter 2 has one section per region): """ + ", ".join(REGIONS) + """
- chapter: 1 Global Outlook, 2 Regional Outlooks, 3 Sovereign Debt Levels and Interest Rates in EMDEs,
  4 Fiscal Policy and Commodity Price Swings
- the Executive Summary, and numbered boxes such as "Box 1.1"

Reply with JSON only:
{"region": "<one region exactly as listed, or null>",
  "chapter": <1, 2, 3, 4 or null>,
  "executive_summary": <true or false>,
  "box": <true if the question names a box, else false>,
  "search_query": "<the question rewritten for search>"}

Rules:
- Set region when the question names a region OR a country in it (India -> South Asia, Brazil -> Latin America and the Caribbean).
- Global or multi-region questions get region null.
- Set chapter only when the question literally says "Chapter N". Never infer it from the topic:
  a wrong filter hides the answer, a missing one only makes search a little slower to find it.
- search_query: spell out acronyms and use the report's words (EMDEs, outlook, projected growth).
  Keep every number and year. Do not add the report's title."""


def understand(question):
    """Question -> filter tags + search query. Unknown values are dropped."""
    reply = get_text(llm.invoke([SystemMessage(content=QUERY_PROMPT), HumanMessage(content=question)]))
    found = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    try:
        data = json.loads(found.group(0)) if found else {}
    except json.JSONDecodeError:
        data = {}
    tags = {}
    if data.get("region") in REGIONS:                       # never trust a name we did not list
        tags["region"] = data["region"]
    if data.get("executive_summary") is True:
        tags["executive_summary"] = True
    elif data.get("chapter") in (1, 2, 3, 4):
        tags["chapter"] = data["chapter"]
    if data.get("box") is True:
        tags["box"] = True
    return tags, (data.get("search_query") or question).strip()


def matches(chunk, tags):
    """Does a child live where the question points? Every child carries chapter, region and kind."""
    if "region" in tags and chunk["region"] != tags["region"]:
        return False
    if tags.get("executive_summary") and chunk["kind"] != "executive_summary":
        return False
    if "chapter" in tags and chunk["chapter_no"] != tags["chapter"]:
        return False
    if tags.get("box") and chunk["kind"] != "box":
        return False
    return True


# ---------------------------------------------------------------------------
# 5. Hybrid search inside the filter: meaning (dense) + exact words (BM25), merged by rank
# ---------------------------------------------------------------------------
def tokenize(text):
    return re.findall(r"\w+", text.lower())


bm25 = BM25Okapi([tokenize(with_heading(c)) for c in children])

RRF_K = 60  # Reciprocal Rank Fusion constant from the original paper


def dense_search(question, n, tags):
    keep = (lambda doc: matches(children[doc.metadata["id"]], tags)) if tags else None
    return [doc.metadata["id"] for doc in vector_store.similarity_search(question, k=n, filter=keep)]


def bm25_search(question, n, tags):
    scores = bm25.get_scores(tokenize(question))
    allowed = [i for i in range(len(children)) if matches(children[i], tags)]
    return sorted(allowed, key=lambda i: scores[i], reverse=True)[:n]


def hybrid_search(question, k=20, tags=None):
    """Each child scores 1/(RRF_K + rank) in every list it appears in; the scores are
    added. Only ranks count, so cosine similarity and BM25 never need a common scale."""
    tags = tags or {}
    fused = {}
    for ranking in (dense_search(question, k, tags), bm25_search(question, k, tags)):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (RRF_K + rank)
    return [children[i] for i in sorted(fused, key=fused.get, reverse=True)[:k]]


def search(question):
    """Understand the question, then hybrid search inside the filter it produced."""
    tags, search_query = understand(question)
    hits = hybrid_search(search_query, tags=tags)
    if len(hits) < 3:                                       # a filter that finds nothing is a bad filter
        hits = hybrid_search(search_query)
    return hits


# ---------------------------------------------------------------------------
# 6. Rerank (Gemini reads the question and the whole shortlist), then parents
# ---------------------------------------------------------------------------
RERANK_PROMPT = (
    "You rank passages by how directly they answer a question.\n"
    "Each passage starts with where it sits in the report (chapter > section > subsection).\n"
    "Reply with JSON only: a list of passage numbers, best first, like [3, 0, 7]. "
    "List at most {k} passages."
)


def rerank(question, hits, k=5):
    numbered = "\n\n".join(f"[{i}] {with_heading(h)}" for i, h in enumerate(hits))
    reply = get_text(llm.invoke([
        SystemMessage(content=RERANK_PROMPT.format(k=k)),
        HumanMessage(content=f"Question: {question}\n\nPassages:\n\n{numbered}"),
    ]))
    found = re.search(r"\[[\d,\s]*\]", reply)
    order = [i for i in json.loads(found.group(0)) if 0 <= i < len(hits)] if found else []
    order = list(dict.fromkeys(order))                       # drop repeats, keep order
    order += [i for i in range(len(hits)) if i not in order]  # fall back to fused order
    return [hits[i] for i in order[:k]]


def to_parents(hits, n=3):
    """Swap each retrieved child for its parent window. Duplicates are dropped; rank order is kept.
    Small children match a question precisely; their ~900-token parent gives the model enough context."""
    out, seen = [], set()
    for h in hits:
        if h["parent_id"] in seen:
            continue
        seen.add(h["parent_id"])
        out.append(parents[h["parent_id"]])
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------------------
# 7. Grounded RAG + memory chatbot
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You answer questions about the World Bank's Global Economic Prospects report (June 2026) "
    "using ONLY the numbered context passages. Each passage shows the chapter and section it comes from.\n"
    "- Use only facts stated in the passages, or already established earlier in this conversation. "
    "Never add outside knowledge.\n"
    "- Only use a passage whose chapter, region or section matches what the question asks about; "
    "never mix up figures from different regions or chapters.\n"
    "- Quote numbers, names and dates exactly as written.\n"
    "- Cite the passages you used with square brackets only, like [1] or [2][3]. Plain text, no bold.\n"
    "- If passages give different figures, report each one with its context.\n"
    "- If the passages do not contain the answer, reply exactly: I could not find this in the document.\n"
    "- Be brief: one to three sentences."
)


def build_context(passages):
    """Number the passages and show where each came from, so the answer can cite them."""
    return "\n\n".join(
        f"[{i}] (pages {p['page_start']}-{p['page_end']}, {p['heading_path']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )


conversation_history = []  # list of HumanMessage / AIMessage, growing each turn


def rag_chat(user_message):
    # 1. Retrieve: understand -> filtered hybrid shortlist of 20 children -> best 5 -> their 3 parents
    passages = to_parents(rerank(user_message, search(user_message)))

    # 2. Build fresh instructions (including the numbered passages) for this turn
    system_message = SystemMessage(content=SYSTEM_PROMPT + "\n\nCONTEXT PASSAGES:\n\n" + build_context(passages))

    # 3. Combine: system instructions + everything said so far + the new message
    messages = [system_message] + conversation_history + [HumanMessage(content=user_message)]
    response = llm.invoke(messages)
    reply_text = get_text(response)

    # 4. Record this turn so future turns remember it
    conversation_history.append(HumanMessage(content=user_message))
    conversation_history.append(AIMessage(content=reply_text))

    return reply_text


# ---------------------------------------------------------------------------
# 8. Interactive loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Global Economic Prospects assistant ready. Type 'quit' to stop.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "quit":
            break
        print("Assistant:", rag_chat(user_input))
