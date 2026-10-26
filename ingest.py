"""
PDF -> heading-tagged parent/child chunks for the World Bank's Global Economic
Prospects (June 2026).

In a long, structured report the words match everywhere ("growth", "outlook");
only the location tells passages apart. So ingestion recovers that location:
  1. Repair the extracted text — ligatures ("ﬂ" -> "fl"), the dropped "Th" glyph
     ("Te conflict" -> "The conflict"), and words hyphenated across a line break.
  2. Read the outline from the fonts — body text is 11pt Garamond, headings are
     Helvetica. Every passage gets its chapter, region, box and heading path
     ("Chapter 2: Regional Outlooks > South Asia > Outlook").
  3. Parents + children — ~900-token parent windows inside one section (what the
     LLM reads) split into ~280-token children (what search matches).

Run on its own to inspect the outline and chunks:
    uv run python ingest.py
"""

import re
from collections import Counter

import pymupdf

CHAPTERS = {
    1: "Global Outlook",
    2: "Regional Outlooks",
    3: "Sovereign Debt Levels and Interest Rates in EMDEs",
    4: "Fiscal Policy and Commodity Price Swings",
}

# Chapter 2 has one section per region, titled exactly like this
REGIONS = [
    "East Asia and Pacific",
    "Europe and Central Asia",
    "Latin America and the Caribbean",
    "Middle East, North Africa, Afghanistan, and Pakistan",
    "South Asia",
    "Sub-Saharan Africa",
]

# Pages with no answers on them: skipped so they never crowd the search results
NOISE_CHAPTERS = {"Front matter", "Summary of Contents", "Contents", "Acknowledgments", "Abbreviations"}

LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}


def tokens(text):
    """Rough token count (~4 characters per token) — good enough for sizing chunks."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# 1. Text repair
# ---------------------------------------------------------------------------
def fix_ligatures(text):
    for glyph, letters in LIGATURES.items():
        text = text.replace(glyph, letters)
    return text


class Repairer:
    """Learns its fixes from the whole document, so no word list is hand-written:
    - "Tese" becomes "These" when "these" is far more common in lower case than "tese"
      (the font loses its "Th" ligature, which only ever starts a capitalised word).
    - "sustainabil-\\nity" loses its hyphen unless "sustainabil-ity" is written on one line elsewhere,
      or the hyphen is suspended ("small- and medium-sized")."""

    def __init__(self, raw_text):
        raw_text = fix_ligatures(raw_text)
        words = Counter(re.findall(r"[A-Za-z]+", raw_text))
        self.th = {w: "Th" + w[1:] for w in words
                   if len(w) > 1 and w[0] == "T" and w[1] != "h"
                   and words["th" + w[1:]] > 5 * words["t" + w[1:]]}
        self.compounds = set(re.findall(r"\w+-\w+", raw_text))

    def __call__(self, text):
        text = fix_ligatures(text)
        text = re.sub(r"(\w+)-\n(\w+)",
                      lambda m: m.group(0).replace("\n", " ") if m.group(2) in ("and", "or", "to")
                      else m.group(0).replace("\n", "") if f"{m.group(1)}-{m.group(2)}" in self.compounds
                      else m.group(1) + m.group(2), text)
        text = re.sub(r"\b[A-Za-z]+\b", lambda m: self.th.get(m.group(0), m.group(0)), text)
        return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 2. Structure from fonts
# ---------------------------------------------------------------------------
def line_kind(text, size, font, bold):
    """This report's typography: body is 11pt Garamond; headings are Helvetica."""
    helvetica = "Helvetica" in font
    if "Andes" in font:
        return "small"                                            # cover pages
    if helvetica and "Cond" in font and size >= 20:
        return "title"                                            # "Executive Summary", "Foreword"
    if re.fullmatch(r"CHAPTER\s+\d", text) and size >= 20:
        return "chapter"                                          # "CHAPTER 2" (not the 7pt running header)
    if size >= 26:
        return "chapter_title"                                    # "REGIONAL OUTLOOKS", "STATISTICAL APPENDIX"
    if re.match(r"^(BOX|TABLE|FIGURE)\s+B?\d", text) and helvetica and bold:
        return "caption"                                          # "BOX 1.1 How much will AI ..."
    if helvetica and bold and size >= 14:
        return "section"                                          # 16pt / 14pt bold: "South Asia", "Global context"
    if helvetica and not bold and abs(size - 11) < 0.3 and len(text) < 90 and not text.endswith("."):
        return "subsection"                                       # "Recent developments", "Outlook"
    if size < 9.2 or helvetica:
        return "small"                                            # chart labels, table cells, footnotes
    return "body"


def page_lines(page):
    """Lines in reading order: wide blocks split the page into bands; inside each band
    the left column is read top to bottom, then the right column."""
    width = page.rect.width
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("lines")]
    ordered, band = [], []

    def flush():
        band.sort(key=lambda b: (b["bbox"][0] >= width / 2, b["bbox"][1]))
        ordered.extend(band)
        band.clear()

    for b in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0])):
        if b["bbox"][2] - b["bbox"][0] > width * 0.55:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()

    for b in ordered:
        for line in b["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            span = max(line["spans"], key=lambda s: len(s["text"].strip()))  # the line's main font
            bold = bool(span["flags"] & 16) or "Bold" in span["font"]
            yield text, line_kind(text, span["size"], span["font"], bold)


def read_sections(path):
    """Walk the report once, tracking where we are. Returns one entry per run of body
    text under the same heading: {chapter, chapter_no, region, box, path, lines: [(text, page)]}."""
    doc = pymupdf.open(path)
    repair = Repairer("".join(page.get_text() for page in doc))
    where = {"chapter": "Front matter", "chapter_no": 0, "section": None, "subsection": None,
             "region": None, "box": None}
    sections, current = [], None
    previous_kind = None

    def heading_path():
        parts = [where["chapter"], where["box"] or where["section"], where["subsection"]]
        return " > ".join(p for p in parts if p)

    for number, page in enumerate(doc, start=1):
        lines = list(page_lines(page))
        if not any(kind == "caption" and text.startswith("BOX") for text, kind in lines):
            if where["box"]:                                      # a box ends where its "(continued)" header stops
                where.update(box=None, subsection=None)
        page_text = []

        for text, kind in lines:
            text = fix_ligatures(text)
            if kind == "title":
                where.update(chapter=text, chapter_no=0, section=None, subsection=None, region=None, box=None)
            elif kind == "chapter":
                n = int(text.split()[-1])
                where.update(chapter=f"Chapter {n}: {CHAPTERS[n]}", chapter_no=n,
                             section=None, subsection=None, region=None, box=None)
            elif kind == "chapter_title" and text == "STATISTICAL APPENDIX":
                where.update(chapter="Statistical Appendix", chapter_no=0,
                             section=None, subsection=None, region=None, box=None)
            elif kind == "section":
                title = f"{where['section']} {text}" if previous_kind == "section" else text  # 2-line headings
                where.update(section=title, subsection=None, box=None,
                             region=title if title in REGIONS else None)
            elif kind == "subsection":
                where["subsection"] = f"{where['subsection']} {text}" if previous_kind == "subsection" else text
            elif kind == "caption" and text.startswith("BOX"):
                title = "Box" + re.sub(r"\s*\(continued\)$", "", text)[3:]
                if title != where["box"]:
                    where.update(box=title, subsection=None)
            elif kind == "body":
                path = heading_path()
                if current is None or current["path"] != path:
                    current = {"chapter": where["chapter"], "chapter_no": where["chapter_no"],
                               "region": where["region"], "box": where["box"] is not None,
                               "path": path, "lines": []}
                    sections.append(current)
                    page_text = []
                if not page_text:
                    current["lines"].append(["", number])
                    page_text = current["lines"][-1]
                page_text[0] += text + "\n"
            previous_kind = kind
        page_text = []                                            # a new page starts a new text run

    for s in sections:
        s["lines"] = [(repair(text), pg) for text, pg in s["lines"]]
    return sections


def section_kind(s):
    if s["chapter"] in NOISE_CHAPTERS or s["path"].endswith("References"):
        return "noise"
    if s["chapter"] == "Executive Summary":
        return "executive_summary"
    return "box" if s["box"] else "body"


# ---------------------------------------------------------------------------
# 3. Parents + children
# ---------------------------------------------------------------------------
ABBREVIATION = re.compile(r"\b(?:U\.S|Mr|Ms|Dr|No|St|Inc|vs|e\.g|i\.e|etc|al|Jan|Feb|Aug|Sept|Oct|Nov|Dec|Fig)\.$")


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])|(?<=[.!?][”\")])\s+(?=[A-Z0-9“\"(])", text)
    sentences = []
    for piece in pieces:
        if sentences and ABBREVIATION.search(sentences[-1]):
            sentences[-1] += " " + piece                          # "U.S. dollar" is one sentence
        else:
            sentences.append(piece)
    return [s for s in sentences if s.strip()]


def group(items, size, overlap=0):
    """Greedy windows of (sentence, page) pairs, ~`size` tokens each, sharing ~`overlap` tokens."""
    windows, current, total = [], [], 0
    for item in items:
        if current and total + tokens(item[0]) > size and total > overlap:
            windows.append(current)
            keep, kept = [], 0
            for prev in reversed(current):
                if kept + tokens(prev[0]) > overlap:
                    break
                keep.insert(0, prev)
                kept += tokens(prev[0])
            current, total = keep, kept
        current.append(item)
        total += tokens(item[0])
    if current and (total > overlap or not windows):
        windows.append(current)
    return windows


def build_hierarchy(sections, child_size=280, parent_size=900):
    """Parents: ~900-token windows inside one section (what the LLM will read).
    Children: ~280-token pieces of a parent (what we search). Every child knows its location."""
    children, parents = [], []
    for s in sections:
        kind = section_kind(s)
        if kind == "noise":
            continue
        sentences = [(x, pg) for text, pg in s["lines"] for x in split_sentences(text)]
        meta = {"chapter": s["chapter"], "chapter_no": s["chapter_no"], "region": s["region"],
                "kind": kind, "heading_path": s["path"]}
        for window in group(sentences, parent_size):
            pid = len(parents)
            parents.append({"text": " ".join(x for x, _ in window), "page_start": window[0][1],
                            "page_end": window[-1][1], "heading_path": s["path"]})
            for piece in group(window, child_size, overlap=40):
                children.append({"id": len(children), "text": " ".join(x for x, _ in piece),
                                 "page_start": piece[0][1], "page_end": piece[-1][1], "parent_id": pid, **meta})
    return children, parents


def with_heading(chunk):
    """What we embed: the heading path travels with the passage."""
    return f"{chunk['heading_path']}\n{chunk['text']}"


def build_chunks(path):
    return build_hierarchy(read_sections(path))


if __name__ == "__main__":
    import os

    pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GEP-Jun-2026.pdf")
    sections = read_sections(pdf)
    children, parents = build_hierarchy(sections)
    print(f"{len(sections)} sections, {len(parents)} parents, {len(children)} children\n")
    for path in dict.fromkeys(s["path"] for s in sections):
        print(" ", path)
    for c in children:
        if c["region"] == "South Asia" and "SAR" in c["text"]:
            print(f"\n--- child {c['id']} (pages {c['page_start']}-{c['page_end']}, region={c['region']})")
            print(with_heading(c))
            break
