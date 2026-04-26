"""국가법령정보 OPEN API 클라이언트.

- lawSearch.do  : 법령 목록 검색 (소관부처 필터)
- lawService.do : 법령 본문 조회 (HTML or JSON)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator

import requests

logger = logging.getLogger(__name__)

LAW_BASE = "https://www.law.go.kr/DRF"


class LawAPIClient:
    def __init__(self, oc: str | None = None, timeout: int = 30) -> None:
        self.oc = oc or os.environ.get("LAW_OC")
        if not self.oc:
            raise RuntimeError("LAW_OC 환경변수 또는 oc 인자가 필요합니다 (예: daro98)")
        self.timeout = timeout
        self.session = requests.Session()

    def search_laws(self, ministry: str, *, display: int = 30,
                    page_max: int = 5, sleep: float = 0.2) -> list[dict]:
        """소관부처(ministry) 기준 법령 검색. 모든 페이지 합쳐 반환."""
        out: list[dict] = []
        for page in range(1, page_max + 1):
            params = {
                "OC": self.oc,
                "target": "law",
                "type": "JSON",
                "query": ministry,
                "display": display,
                "page": page,
            }
            r = self.session.get(f"{LAW_BASE}/lawSearch.do", params=params, timeout=self.timeout)
            r.raise_for_status()
            payload = r.json()
            laws = payload.get("LawSearch", {}).get("law", [])
            if not laws:
                break
            if isinstance(laws, dict):
                laws = [laws]
            kept = [law for law in laws if law.get("소관부처명") == ministry]
            out.extend(kept)
            if len(laws) < display:
                break
            time.sleep(sleep)
        return out

    def fetch_law_html(self, mst: str) -> str:
        """법령 본문 HTML."""
        params = {
            "OC": self.oc,
            "target": "law",
            "MST": mst,
            "type": "HTML",
        }
        r = self.session.get(f"{LAW_BASE}/lawService.do", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.text

    def fetch_law_json(self, mst: str) -> dict:
        """법령 본문 JSON (구조화)."""
        params = {
            "OC": self.oc,
            "target": "law",
            "MST": mst,
            "type": "JSON",
        }
        r = self.session.get(f"{LAW_BASE}/lawService.do", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def iter_laws(self, ministries: list[str], *, max_per_ministry: int = 30,
                  fetch_body: bool = True, sleep: float = 0.3) -> Iterator[dict]:
        for ministry in ministries:
            laws = self.search_laws(ministry)
            for law in laws[:max_per_ministry]:
                mst = law.get("법령일련번호")
                if not mst:
                    continue
                rec = dict(law)
                if fetch_body:
                    try:
                        rec["body_json"] = self.fetch_law_json(str(mst))
                    except Exception as e:  # noqa: BLE001
                        logger.warning("body fetch fail mst=%s: %s", mst, e)
                        rec["body_json"] = None
                yield rec
                time.sleep(sleep)
