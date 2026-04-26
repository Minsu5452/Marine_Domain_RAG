"""법령 JSON → 조문 단위 정규화.

국가법령정보 lawService.do JSON 응답은 법령 메타 + 조문 트리(조 > 항 > 호)로 구성된다.
응답 스키마는 법령마다 유연하므로 방어적으로 파싱한다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _flatten(obj: Any) -> list[Any]:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    return [obj]


def _coerce_text(node: Any) -> str:
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, list):
        return " ".join(_coerce_text(x) for x in node)
    if isinstance(node, dict):
        # 흔한 텍스트 키들
        for k in ("조문내용", "항내용", "호내용", "조문제목", "내용", "value", "@value"):
            if k in node:
                return _coerce_text(node[k])
        return " ".join(_coerce_text(v) for v in node.values())
    return str(node)


def parse_law(record: dict) -> list[dict]:
    """단일 법령 record → 조문 row 리스트."""
    body = record.get("body_json") or {}
    law_root = body.get("법령") or body.get("LawService") or body
    basic = law_root.get("기본정보") or {}
    arts = (law_root.get("조문") or {}).get("조문단위") or []
    arts = _flatten(arts)

    law_name = (
        basic.get("법령명_한글")
        or basic.get("법령명한글")
        or record.get("법령명한글")
        or record.get("법령명_한글")
    )
    law_id = basic.get("법령ID") or record.get("법령ID") or record.get("법령일련번호")
    ministry = (
        (basic.get("소관부처") or {}).get("content")
        if isinstance(basic.get("소관부처"), dict)
        else basic.get("소관부처명") or record.get("소관부처명")
    )

    rows: list[dict] = []
    for art in arts:
        if not isinstance(art, dict):
            continue
        art_no = art.get("조문번호") or ""
        art_title = _coerce_text(art.get("조문제목"))
        art_text = _coerce_text(art.get("조문내용"))
        sub_items = _flatten(art.get("항"))
        sub_text_parts = [_coerce_text(s) for s in sub_items]
        full_text = "\n".join([t for t in [art_title, art_text, *sub_text_parts] if t])
        if not full_text.strip():
            continue
        rows.append({
            "law_id": str(law_id) if law_id is not None else "",
            "law_name": str(law_name or ""),
            "ministry": str(ministry or ""),
            "article_no": str(art_no),
            "article_title": str(art_title),
            "text": full_text,
        })
    return rows


def parse_laws(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for rec in records:
        try:
            rows.extend(parse_law(rec))
        except Exception as e:  # noqa: BLE001
            logger.warning("parse fail mst=%s: %s", rec.get("법령일련번호"), e)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["doc_id"] = df["law_id"].astype(str) + "::" + df["article_no"].astype(str)
        df = df.drop_duplicates("doc_id").reset_index(drop=True)
    return df
