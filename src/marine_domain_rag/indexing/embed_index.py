"""한국어 sentence-transformers + FAISS 인덱스 + BM25 hybrid retriever."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class HybridRetriever:
    embedder: SentenceTransformer
    faiss_index: faiss.Index
    bm25: BM25Okapi
    articles: pd.DataFrame
    bm25_weight: float
    embed_weight: float

    def search(self, query: str, k: int = 5) -> pd.DataFrame:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.faiss_index.search(np.asarray(q_emb, dtype="float32"), k * 4)
        embed_scores = np.zeros(len(self.articles), dtype="float32")
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(embed_scores):
                embed_scores[idx] = float(score)

        tokens = _tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        combined = self.embed_weight * embed_scores + self.bm25_weight * bm25_scores

        top_idx = np.argsort(-combined)[:k]
        out = self.articles.iloc[top_idx].copy()
        out["score"] = combined[top_idx]
        return out.reset_index(drop=True)


def _tokenize(text: str) -> list[str]:
    return [t for t in text.replace("\n", " ").split() if t.strip()]


def build_index(articles: pd.DataFrame, model_name: str, batch_size: int,
                max_seq_length: int, out_dir: str | Path,
                bm25_weight: float, embed_weight: float) -> HybridRetriever:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embedder = SentenceTransformer(model_name)
    embedder.max_seq_length = max_seq_length

    texts = (articles["law_name"].fillna("") + " " + articles["article_title"].fillna("")
             + " " + articles["text"].fillna("")).tolist()
    embeds = embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=True)
    embeds = np.asarray(embeds, dtype="float32")
    dim = embeds.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeds)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeds.npy", embeds)

    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    articles.to_parquet(out_dir / "articles.parquet")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "bm25_weight": bm25_weight,
            "embed_weight": embed_weight,
            "n_docs": int(len(articles)),
            "dim": int(dim),
        }, f, ensure_ascii=False, indent=2)
    return HybridRetriever(embedder, index, bm25, articles, bm25_weight, embed_weight)


def load_index(out_dir: str | Path, bm25_weight: float | None = None,
               embed_weight: float | None = None,
               device: str = "cpu") -> HybridRetriever:
    """Load 시 sbert device 기본값을 cpu 로 둔다 (LLM 과 동시 메모리 압박 회피)."""
    out_dir = Path(out_dir)
    with open(out_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedder = SentenceTransformer(meta["model_name"], device=device)
    embedder.max_seq_length = int(meta["max_seq_length"])
    index = faiss.read_index(str(out_dir / "faiss.index"))
    articles = pd.read_parquet(out_dir / "articles.parquet")
    texts = (articles["law_name"].fillna("") + " " + articles["article_title"].fillna("")
             + " " + articles["text"].fillna("")).tolist()
    bm25 = BM25Okapi([_tokenize(t) for t in texts])
    return HybridRetriever(
        embedder=embedder,
        faiss_index=index,
        bm25=bm25,
        articles=articles,
        bm25_weight=bm25_weight if bm25_weight is not None else float(meta.get("bm25_weight", 0.4)),
        embed_weight=embed_weight if embed_weight is not None else float(meta.get("embed_weight", 0.6)),
    )
