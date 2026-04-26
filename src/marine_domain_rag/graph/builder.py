"""GraphRAG 빌더.

노드: article(조문) + term(빈출 명사 후보)
엣지:
  article -> term      (포함)
  term <-> term        (같은 조문 내 공출현)
  article -> article   (같은 법령, 인접 조문)
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def _candidate_terms(text: str, *, min_len: int = 2, max_len: int = 12) -> list[str]:
    """공백 분리 + 한국어 어미 단순 stripping. 운영에서는 KoNLPy 권장."""
    out = []
    for tok in text.replace("\n", " ").split():
        tok = tok.strip("·,.()[]{}\"'·、，。．「」『』")
        if min_len <= len(tok) <= max_len:
            out.append(tok)
    return out


def build_graph(articles: pd.DataFrame, *, min_term_freq: int = 3,
                top_terms_per_article: int = 8) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    term_counter: Counter[str] = Counter()
    article_terms: dict[str, list[str]] = {}
    for _, r in articles.iterrows():
        toks = _candidate_terms(r["text"])
        term_counter.update(toks)
        article_terms[r["doc_id"]] = toks

    valid_terms = {t for t, c in term_counter.items() if c >= min_term_freq}

    for _, r in articles.iterrows():
        node_id = f"art::{r['doc_id']}"
        g.add_node(node_id, kind="article", law_id=r["law_id"], law_name=r["law_name"],
                   article_no=r["article_no"], article_title=r["article_title"])

    for doc_id, toks in article_terms.items():
        node_art = f"art::{doc_id}"
        toks_filtered = [t for t in toks if t in valid_terms]
        c = Counter(toks_filtered)
        for term, cnt in c.most_common(top_terms_per_article):
            node_term = f"term::{term}"
            if not g.has_node(node_term):
                g.add_node(node_term, kind="term", text=term)
            g.add_edge(node_art, node_term, type="contains", weight=cnt)
            g.add_edge(node_term, node_art, type="appears_in", weight=cnt)

    for _, group in articles.groupby("law_id"):
        ordered = group.sort_values("article_no")
        ids = ordered["doc_id"].tolist()
        for a, b in zip(ids, ids[1:]):
            g.add_edge(f"art::{a}", f"art::{b}", type="next_in_law", weight=1)
            g.add_edge(f"art::{b}", f"art::{a}", type="prev_in_law", weight=1)

    for doc_id, toks in article_terms.items():
        toks_set = list({t for t in toks if t in valid_terms})
        for a, b in combinations(toks_set, 2):
            na, nb = f"term::{a}", f"term::{b}"
            if g.has_node(na) and g.has_node(nb):
                g.add_edge(na, nb, type="cooccurs", weight=1)

    return g


def save_graph(g: nx.MultiDiGraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(g, f)


def load_graph(path: str | Path) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


def expand_via_graph(g: nx.MultiDiGraph, seed_doc_ids: list[str], *, hops: int = 2,
                     max_articles: int = 20) -> list[str]:
    """seed 조문에서 그래프를 통해 관련 조문 ID 확장."""
    found: dict[str, int] = {}
    frontier = [f"art::{x}" for x in seed_doc_ids]
    visited = set(frontier)
    for hop in range(hops):
        next_frontier = []
        for node in frontier:
            for nbr in g.successors(node):
                if g.nodes[nbr].get("kind") == "article" and nbr not in visited:
                    found[nbr] = found.get(nbr, 0) + 1
                    visited.add(nbr)
                    next_frontier.append(nbr)
                elif g.nodes[nbr].get("kind") == "term":
                    for nbr2 in g.successors(nbr):
                        if (g.nodes[nbr2].get("kind") == "article"
                                and nbr2 not in visited):
                            found[nbr2] = found.get(nbr2, 0) + 1
                            visited.add(nbr2)
        frontier = next_frontier
    ranked = sorted(found.items(), key=lambda kv: -kv[1])
    return [n.removeprefix("art::") for n, _ in ranked[:max_articles]]
