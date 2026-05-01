"""
pipeline/retriever.py
---------------------
BM25 corpus retriever. Indexes every markdown file under data/
and returns the top-k most relevant passages for a given query.

Zero external dependencies — pure Python stdlib.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single retrievable passage from the corpus."""
    doc_id: str          # relative path from data root
    company: str         # hackerrank | claude | visa
    category: str        # sub-directory category (raw)
    text: str            # raw markdown content
    title: str = ""      # first heading or filename stem


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Text tokenisation
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "of", "and", "or", "to", "i",
    "my", "we", "our", "for", "on", "this", "that", "with", "at", "by",
    "from", "as", "be", "was", "are", "have", "has", "had", "do", "did",
    "can", "will", "would", "not", "but", "so", "if", "how", "what",
    "when", "where", "who", "which", "me", "you", "your", "us", "them",
    "their", "its", "no", "yes", "please", "help", "want", "need", "get",
}


def tokenise(text: str) -> list[str]:
    """Lowercase, strip markdown noise, split on non-alphanumeric."""
    text = text.lower()
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # strip links
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # strip code
    text = re.sub(r"`[^`]*`", " ", text)                     # strip inline code
    tokens = re.findall(r"[a-z0-9']+", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _extract_title(text: str, fallback: str) -> str:
    m = re.search(r"^#+\s+(.+)", text, re.MULTILINE)
    return m.group(1).strip() if m else fallback


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def load_corpus(data_root: Path) -> list[Chunk]:
    """Walk data_root and load every .md file as a Chunk."""
    chunks: list[Chunk] = []
    for company_dir in sorted(data_root.iterdir()):
        if not company_dir.is_dir():
            continue
        company = company_dir.name.lower()
        for md_path in sorted(company_dir.rglob("*.md")):
            rel = str(md_path.relative_to(data_root))
            try:
                text = md_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if not text.strip():
                continue
            # Derive category from directory structure
            rel_parts = md_path.relative_to(data_root).parts
            category = rel_parts[1] if len(rel_parts) >= 2 else company
            title = _extract_title(text, md_path.stem)
            chunks.append(Chunk(
                doc_id=rel,
                company=company,
                category=category,
                text=text,
                title=title,
            ))
    return chunks


# ---------------------------------------------------------------------------
# BM25  (Okapi BM25 — Robertson et al.)
# ---------------------------------------------------------------------------

class BM25:
    """Lightweight BM25 index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._N: int = 0

    def fit(self, tokenised_docs: list[list[str]]) -> None:
        self._corpus = tokenised_docs
        self._N = len(tokenised_docs)
        total_len = 0
        df: dict[str, int] = {}
        for doc in tokenised_docs:
            total_len += len(doc)
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._avgdl = total_len / max(self._N, 1)

    def _score_doc(self, query_tokens: list[str], doc_idx: int) -> float:
        doc = self._corpus[doc_idx]
        dl = len(doc)
        tf_map: dict[str, int] = {}
        for t in doc:
            tf_map[t] = tf_map.get(t, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in self._df:
                continue
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            df = self._df[term]
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
            numer = tf * (self.k1 + 1)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * (numer / denom)
        return score

    def rank(self, query_tokens: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        if not query_tokens:
            return []
        scores = [(i, self._score_doc(query_tokens, i)) for i in range(self._N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ---------------------------------------------------------------------------
# Retriever  (public API)
# ---------------------------------------------------------------------------

class CorpusRetriever:
    """
    Load corpus once → search many times.
    Optionally filters by company for higher precision.
    """

    def __init__(self, data_root: str | Path) -> None:
        self._data_root = Path(data_root)
        self._chunks: list[Chunk] = []
        self._bm25 = BM25()
        self._tokenised: list[list[str]] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._chunks = load_corpus(self._data_root)
        self._tokenised = [tokenise(c.text) for c in self._chunks]
        self._bm25.fit(self._tokenised)
        self._loaded = True

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
        expand_k: int = 30,
    ) -> list[SearchResult]:
        """
        Retrieve the top_k most relevant corpus chunks.

        If company is provided, filter to that company's documents
        but fall back to global results if too few matches.
        """
        if not self._loaded:
            self.load()

        q_tokens = tokenise(query)
        if not q_tokens:
            return []

        # Get a larger candidate pool, then filter
        candidates = self._bm25.rank(q_tokens, top_k=expand_k)

        results: list[SearchResult] = []
        for idx, score in candidates:
            chunk = self._chunks[idx]
            if company and chunk.company != company.lower():
                continue
            results.append(SearchResult(chunk=chunk, score=score))
            if len(results) >= top_k:
                break

        # If company filter was too restrictive, add global results
        if company and len(results) < 2:
            seen = {r.chunk.doc_id for r in results}
            for idx, score in candidates:
                chunk = self._chunks[idx]
                if chunk.doc_id not in seen:
                    results.append(SearchResult(chunk=chunk, score=score))
                    seen.add(chunk.doc_id)
                if len(results) >= top_k:
                    break

        return results
