# Global Economic Prospects (June 2026) — Q&A (Advanced Document RAG)

Use Case 2 of the RAG hands-on exercises: a question-answering chatbot over the World
Bank's Global Economic Prospects, June 2026 (200 pages). Built with LangChain + Google
Gemini (+ local fastembed embeddings) on the same structure as `chatbot-pilot`, with both a CLI and a browser UI.

The report has four chapters, boxed articles and six regions that all talk about
"growth forecasts". The words match everywhere, so the location of a passage is what
tells it apart. The pipeline keeps track of where every passage lives and searches there.

## How it works

```
Offline (once):  PDF -> Text repair -> Outline from fonts -> Parents + children
                     -> local bge-small embeddings of "heading path + text" (+ BM25) -> .index/
Every question:  Question -> Gemini query understanding (region / chapter filter + rewrite)
                     -> Hybrid search inside the filter -> Gemini rerank (5 children)
                     -> Parent sections (3) -> Cited answer
```

| Problem with basic RAG | Fix |
|---|---|
| "Te conﬂict", "sustainabil- ity" | Repair the text: ligatures, the dropped "Th" glyph, words hyphenated across lines |
| No idea which chapter or region a passage is from | Read headings from the fonts (Helvetica headings, Garamond body): every chunk carries chapter, region, box and heading path |
| Passage says "SAR", not "South Asia" | Embed the heading path ("Chapter 2: Regional Outlooks > South Asia > Outlook") with the text |
| Question names a place or chapter | One Gemini call turns it into a metadata filter plus a search-friendly rewrite (India -> South Asia); if the filter finds too little, search the whole report |
| Small chunks lack context | Search ~280-token children, hand the model their ~900-token parent section |

## Project structure

```
ingest.py       # PDF -> chunks: text repair, outline from fonts, parents + children
chatbot.py      # embeddings, query understanding, filtered hybrid search, rerank, parents, rag_chat()
server.py       # FastAPI bridge exposing rag_chat() to the browser UI
static/         # HTML/CSS/JS frontend (index.html, style.css, script.js)
data/           # GEP-Jun-2026.pdf, the source document
pyproject.toml  # dependencies (managed with uv)
```

The first run chunks and embeds the PDF (a few hundred chunks, so it takes a moment) and
caches the result in `.index/`. Later runs load it from there. Delete `.index/` to rebuild.

Embeddings run locally with [fastembed](https://github.com/qdrant/fastembed)
(`BAAI/bge-small-en-v1.5`, CPU, no API quota). The model (~67 MB) downloads once on
first run. Gemini is used only for answering, reranking and reading the question.

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) installed
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup

```bash
uv sync
```

## Running it

### Option A — command-line chatbot

```bash
uv run python chatbot.py
```

You'll be prompted once for your Gemini API key (input is hidden, never saved to
disk). Chat in the terminal; type `quit` to exit.

### Option B — browser UI

```bash
uv run uvicorn server:app --reload
```

Then open **http://127.0.0.1:8000/**. The first request triggers the same API-key
prompt in the terminal running the server. Use the **Reset** button to clear
conversation memory.

To skip the prompt, export the key first: `export GOOGLE_API_KEY="your-key-here"`.

### Inspecting the outline and chunks

```bash
uv run python ingest.py
```

Prints every heading path found in the report and a sample child chunk, without calling Gemini.

## Expected outcome

The chatbot should answer these correctly, from the right chapter and section, without
mixing up regions or chapters:

- What is the growth outlook for South Asia in this report?
- What does Chapter 3 say about sovereign debt in developing economies?
- According to the executive summary, what are the main risks to the global outlook?
