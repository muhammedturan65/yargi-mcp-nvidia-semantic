"""
Token-aware chunker — Türk hukuki karar metinleri için.

NVIDIA nv-embed-v1'in 5120 token context limitine uymak için metni ~512-token
chunk'lara böler. Mevcut `semantic_search/processor.py` character-based; bu
modül token-based ve hukuki yapıya duyarlıdır.

Strateji:
  1. Yargıtay/Danıştay kararlarında tipik yapı:
       - Başlık (Daire, Esas/Karar No, Tarih)
       - "Gerekçe" / "TURKISH" / "KARAR" bölümleri
       - Hüküm
     Her bölüm ayrı bir chunk başlangıcı olur.
  2. Cümle sınırlarında böl (paragraph'ı koru).
  3. Chunk başına ~400-512 token hedefle, 80 token overlap ile.
  4. Her chunk'a parent metadata ekle (document_id, esas_no, vb).

Token sayısı tiktoken (cl100k_base — Llama 3 ile uyumlu) ile hesaplanır.
tiktoken yoksa karakter-bazlı fallback (4 char ≈ 1 token).
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# tiktoken opsiyonel — yoksa char-based fallback
try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text, disallowed_special=()))

    _TOKENIZER_NAME = "tiktoken/cl100k_base"

except Exception:
    # Fallback: ~4 char/token (Llama 3 tokenizer'ına yaklaşık)
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    _TOKENIZER_NAME = "char-approx(4char=1token)"

logger.info(f"Chunker tokenizer: {_TOKENIZER_NAME}")


# Yargıtay kararlarında sık görülen bölüm başlıkları
SECTION_PATTERNS = [
    re.compile(r"^\s*(G\s*[Eİ]\s*R\s*[EÇ]\s*K\s*[ÇC]\s*E\s*[:\.]?)", re.IGNORECASE),
    re.compile(r"^\s*(T\s*U\s*R\s*K\s*[İI]\s*[SŞ]\s*[:\.]?)", re.IGNORECASE),
    re.compile(r"^\s*(K\s*A\s*R\s*A\s*R\s*[:\.]?)", re.IGNORECASE),
    re.compile(r"^\s*(H\s*[ÜU]\s*K\s*[ÜU]\s*M\s*[:\.]?)", re.IGNORECASE),
    re.compile(r"^\s*(DAVACI|DAVALI|İHBAR OLUNAN)\s*[:\.]?", re.IGNORECASE),
    re.compile(r"^\s*(Ö\s*Z\s*E\s*T\s*[:\.]?)", re.IGNORECASE),
    re.compile(r"^\s*(B\s*A\s*[SŞ]\s*L\s*I\s*K\s*[:\.]?)", re.IGNORECASE),
]

# Cümle sonu pattern (Türkçe kısaltmaları göz ardı etmeye çalışır)
_ABBREV = re.compile(
    r"\b(?:Dr|Prof|Doç|Yrd|Av|Md|Sr|No|Sn|T|bl|vd|vs|vb|bkz|gm|gn)\.\s",
    re.IGNORECASE,
)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÇĞİÖŞÜ])")


@dataclass
class Chunk:
    """Bir belgenin bir parçası."""
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    section: Optional[str] = None  # GEREKÇE, KARAR, HÜKÜM, vb.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "token_count": self.token_count,
            "section": self.section,
        }


class LegalChunker:
    """
    Hukuki belgeler için token-aware chunker.

    Args:
        target_tokens: Hedef chunk boyutu (token). Default 512 (nv-embed-v1 için güvenli).
        overlap_tokens: Çakışma miktarı (token). Default 80.
        min_tokens: Alt limit (altındakiler komşu chunk'a merge edilir). Default 50.
        max_tokens: Üst limit (token sayısı aşılırsa zorla böl). Default 700.
    """

    def __init__(
        self,
        target_tokens: int = 512,
        overlap_tokens: int = 80,
        min_tokens: int = 50,
        max_tokens: int = 700,
    ):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        if overlap_tokens >= target_tokens:
            raise ValueError("overlap_tokens, target_tokens'tan küçük olmalı")

        logger.info(
            f"LegalChunker init: target={target_tokens}, overlap={overlap_tokens}, "
            f"min={min_tokens}, max={max_tokens}"
        )

    def chunk_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Belgeyi chunk'lara böl.

        Args:
            document_id: Belge ID'si
            text: Belge metni
            metadata: Belge-level metadata (esas_no, karar_no, vb.)

        Returns:
            List[Chunk] — her chunk belge-level metadata'nın kopyasını taşır
                         + chunk_index, total_chunks, section bilgisi
        """
        if not text or not text.strip():
            return []

        base_meta = dict(metadata or {})

        # 1. Metni temizle
        cleaned = self._clean_text(text)

        # 2. Bölümlere ayır (GEREKÇE, KARAR, vb.) — bölüm başına chunk listesi
        sections = self._split_by_sections(cleaned)

        # 3. Her bölümü cümlelere böl
        all_units: List[Dict[str, Any]] = []  # [{"text": ..., "section": ..., "tokens": ...}, ...]
        for section_name, section_text in sections:
            sentences = self._split_sentences(section_text)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                tok = count_tokens(sent)
                if tok == 0:
                    continue
                all_units.append({
                    "text": sent,
                    "section": section_name,
                    "tokens": tok,
                })

        if not all_units:
            return []

        # 4. Cümleleri chunk'lara topla (target_tokens hedef, max_tokens üst sınır)
        raw_chunks: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0
        current_section: Optional[str] = None

        for unit in all_units:
            # Bölüm değiştiyse yeni chunk başlat
            if current_section != unit["section"] and current:
                raw_chunks.append({
                    "text": " ".join(u["text"] for u in current),
                    "section": current_section,
                    "tokens": current_tokens,
                    "units": current,
                })
                # Overlap: son cümleleri yeni chunk'a taşı
                overlap_units, overlap_tokens = self._take_overlap(current)
                current = overlap_units
                current_tokens = overlap_tokens
                current_section = unit["section"]

            # Cümle ekle — max_tokens aşarsa chunk'ı kapat
            if current and current_tokens + unit["tokens"] > self.max_tokens:
                raw_chunks.append({
                    "text": " ".join(u["text"] for u in current),
                    "section": current_section,
                    "tokens": current_tokens,
                    "units": current,
                })
                overlap_units, overlap_tokens = self._take_overlap(current)
                current = overlap_units
                current_tokens = overlap_tokens

            current.append(unit)
            current_tokens += unit["tokens"]
            if current_section is None:
                current_section = unit["section"]

        # Son chunk
        if current:
            raw_chunks.append({
                "text": " ".join(u["text"] for u in current),
                "section": current_section,
                "tokens": current_tokens,
                "units": current,
            })

        # 5. Çok küçük chunk'ları öncekiyle birleştir
        merged = self._merge_small(raw_chunks)

        # 6. Chunk objelerine dönüştür
        total = len(merged)
        chunks: List[Chunk] = []
        for idx, rc in enumerate(merged):
            chunk_id = self._gen_chunk_id(document_id, idx)
            chunk_meta = {
                **base_meta,
                "chunk_index": idx,
                "total_chunks": total,
                "section": rc["section"] or "body",
                "token_count": rc["tokens"],
            }
            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=rc["text"],
                metadata=chunk_meta,
                chunk_index=idx,
                total_chunks=total,
                token_count=rc["tokens"],
                section=rc["section"],
            ))

        logger.info(
            f"Belge {document_id} {total} chunk'a bölündü "
            f"(toplam {sum(c.token_count for c in chunks)} token, "
            f"avg {sum(c.token_count for c in chunks)//max(len(chunks),1)} token/chunk)"
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_text(self, text: str) -> str:
        """Aşırı whitespace'i temizle, Türkçe karakterleri koru."""
        # UTF-8 BOM
        text = text.lstrip("\ufeff")
        # Multiple whitespace → single space (newlines dahil ama paragraph'lardan önce sakla)
        # Önce newlines'ları koruyarak clean, sonra normalize
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_by_sections(self, text: str) -> List[tuple]:
        """
        Karar metnini bölümlere ayır.
        Dönüş: [(section_name, section_text), ...]

        Bölüm başlığı tespit edilemezse tek bir "body" bölümü döner.
        """
        lines = text.split("\n")
        sections: List[tuple] = []
        current_section = "body"
        current_lines: List[str] = []

        for line in lines:
            line_stripped = line.strip()
            matched_section = None
            if line_stripped and len(line_stripped) < 80:
                for pat in SECTION_PATTERNS:
                    if pat.match(line_stripped):
                        # Matchtan bölüm adını çıkar
                        matched_section = line_stripped.rstrip(":.").upper()
                        break

            if matched_section is not None:
                # Yeni bölüm başladı — eskisini kaydet
                if current_lines:
                    sections.append((current_section, "\n".join(current_lines)))
                # Bölüm adını kısalt: sadece ilk kelime (GEREKÇE, HÜKÜM, ÖZET, vb.)
                # "GEREKÇE: DAVA, MURIS..." → "GEREKÇE"
                short_section = re.split(r"[\s:]", matched_section, 1)[0].strip()
                current_section = short_section or matched_section
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_section, "\n".join(current_lines)))

        # Boş bölümleri ele
        return [(name, txt) for name, txt in sections if txt.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """
        Türkçe metni cümlelere böl.
        Kısaltmaları (Dr., Prof., Av.) ve sayı noktalarını korur.
        """
        # Kısaltmalardaki noktaları geçici olarak değiştir
        temp = text
        abbrev_replacements: List[tuple] = []

        def _replace_abbrev(m):
            placeholder = f"__ABBR{len(abbrev_replacements)}__"
            abbrev_replacements.append((placeholder, m.group(0)))
            return placeholder

        temp = _ABBREV.sub(_replace_abbrev, temp)

        # Sayı içindeki noktaları koru: "1.5" → "1__DOT__5"
        temp = re.sub(r"(\d)\.(\d)", r"\1__DOT__\2", temp)

        # Tarihi koru: "12.05.2024" → zaten yukarıda yakalanır (3 tane nokta)
        # Tarih pattern'i: dd.mm.yyyy
        temp = re.sub(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", r"\1__DT__\2__DT__\3", temp)

        # Şimdi cümleleri böl
        sentences = _SENT_END.split(temp)

        # Placeholder'ları geri ver
        cleaned: List[str] = []
        for sent in sentences:
            for placeholder, original in abbrev_replacements:
                sent = sent.replace(placeholder, original)
            sent = sent.replace("__DOT__", ".").replace("__DT__", ".")
            sent = sent.strip()
            if sent and len(sent) > 3:
                cleaned.append(sent)

        return cleaned

    def _take_overlap(self, units: List[Dict[str, Any]]) -> tuple:
        """Son birkaç cümleyi overlap olarak al (overlap_tokens hedefine ulaşana kadar)."""
        if not units:
            return [], 0
        overlap: List[Dict[str, Any]] = []
        total = 0
        for u in reversed(units):
            if total + u["tokens"] > self.overlap_tokens:
                break
            overlap.insert(0, u)
            total += u["tokens"]
        return overlap, total

    def _merge_small(self, raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """min_tokens altındaki chunk'ları önceki ile birleştir."""
        if len(raw_chunks) <= 1:
            return raw_chunks

        merged: List[Dict[str, Any]] = []
        for rc in raw_chunks:
            if (merged and rc["tokens"] < self.min_tokens
                    and merged[-1]["tokens"] + rc["tokens"] <= self.max_tokens):
                # Birleştir
                prev = merged[-1]
                prev["text"] = prev["text"] + " " + rc["text"]
                prev["tokens"] += rc["tokens"]
                prev["units"].extend(rc["units"])
                # Section bilgisini koru (önceki)
            else:
                merged.append(dict(rc))
        return merged

    def _gen_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Deterministic chunk ID — aynı belge her zaman aynı ID'leri üretir."""
        h = hashlib.md5(f"{document_id}__c{chunk_index}".encode()).hexdigest()[:10]
        return f"{document_id}__c{chunk_index}_{h}"


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------

def chunk_text(
    document_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    target_tokens: int = 512,
    overlap_tokens: int = 80,
) -> List[Chunk]:
    """Quick helper: metni tek çağrıda chunk'lara böl."""
    chunker = LegalChunker(
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    return chunker.chunk_document(document_id, text, metadata=metadata)
