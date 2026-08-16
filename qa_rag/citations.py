"""
Atıf (citation) formatlama yardımcıları.

LLM'in ürettiği cevabın sonuna kaynak listesi ekler. Format:
    [1] Yargıtay 7. HD, E.2023/1234, K.2023/5678 (15.03.2023)
    [2] Yargıtay 3. HD, E.2022/9876, K.2022/5432 (10.05.2022)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Citation:
    """Tek bir hukuki karar atfı."""
    index: int                  # [1], [2], ...
    birim_adi: str              # Yargıtay 7. Hukuk Dairesi
    esas_no: Optional[str]      # 2023/1234
    karar_no: Optional[str]     # 2023/5678
    karar_tarihi: Optional[str] # 15.03.2023
    document_id: Optional[str]  # Bedesten/internal document ID
    similarity_score: float     # RAG retrieval skoru (0-1)
    source_url: Optional[str] = None

    def format(self) -> str:
        """İnsan-okunabilir atıf metni üret."""
        parts = [f"[{self.index}]"]

        if self.birim_adi:
            parts.append(self.birim_adi)
        if self.esas_no:
            parts.append(f"E.{self.esas_no}")
        if self.karar_no:
            parts.append(f"K.{self.karar_no}")
        if self.karar_tarihi:
            parts.append(f"({self.karar_tarihi})")

        cite_str = ", ".join(parts[1:]) if len(parts) > 1 else ""
        return f"{parts[0]} {cite_str}".strip()


def build_citations_from_decisions(decisions: List[Dict]) -> List[Citation]:
    """
    RAG sonuç listesinden Citation nesneleri üret.

    Args:
        decisions: search_bedesten_semantic'in results listesi (formatted_results)
                   veya RAGContext.decisions

    Returns:
        List[Citation] — sıralı (1-indexed)
    """
    citations = []
    for i, d in enumerate(decisions, 1):
        meta = d.get("metadata", {})
        citations.append(Citation(
            index=i,
            birim_adi=meta.get("birim_adi", d.get("title", "Bilinmeyen Daire").split(" - ")[0]),
            esas_no=meta.get("esas_no"),
            karar_no=meta.get("karar_no"),
            karar_tarihi=meta.get("karar_tarihi"),
            document_id=d.get("document_id"),
            similarity_score=float(d.get("similarity_score", d.get("score", 0))),
            source_url=d.get("source_url"),
        ))
    return citations


def format_citations(citations: List[Citation], include_scores: bool = False) -> str:
    """
    Atıf listesini okunabilir metin olarak üret.

    Args:
        citations: List[Citation]
        include_scores: True ise her atıftan sonra benzerlik skorunu ekle

    Returns:
        "Kaynaklar:\n[1] ...\n[2] ...\n..." formatında metin
    """
    if not citations:
        return "Kaynaklar: (bulunamadı)"

    lines = ["Kaynaklar:", ""]
    for c in citations:
        line = c.format()
        if include_scores:
            line += f"  [benzerlik: {c.similarity_score:.4f}]"
        if c.source_url:
            line += f"\n    URL: {c.source_url}"
        lines.append(line)

    return "\n".join(lines)
