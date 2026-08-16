"""
qa_rag/indexer.py — Bedesten → ChromaDB index pipeline.

Bu modül, Bedesten API'den Türk hukuki kararların TAM METNİNİ çeker,
token-aware chunker ile parçalara böler, NVIDIA nv-embed-v1 ile embed'ler
ve ChromaDB'ye kalıcı olarak yazar.

Bu, rag_engine'in "load_corpora" adımının yerine geçer:
  - Eski (v1.1.0): Her sorguda 30 belge çek, in-memory store'a koy, 70s bekle
  - Yeni (v1.2.0): Bir kez 1000+ belgeyi ChromaDB'ye indexle, sonradan 50ms

Pipeline:
  1. Bedesten search (keyword) → decision ID listesi
  2. Her decision ID → get_document_as_markdown (TAM METİN)
  3. LegalChunker ile ~512-token chunk'lara böl
  4. NVIDIA nv-embed-v1 ile batch embed (passage input_type)
  5. ChromaVectorStore.add_chunks() ile kalıcı yaz
  6. Progress + checkpoint (kesinti olursa kaldığı yerden devam)

Env vars (Mevcut MCP env'leri +):
  INDEXER_BATCH_SIZE     — NVIDIA'ya bir seferde kaç chunk embed'lenecek (default 32)
  INDEXER_TARGET_DOCS    — Hedef belge sayısı (default 200)
  INDEXER_KEYWORDS       — Virgülle ayrılmış anahtar kelimeler (default: muvazaa,tapu,iptal)
  INDEXER_COURT_TYPES    — Virgülle ayrılmış mahkeme tipleri (default: YARGITAYKARARI)
  CHROMA_PERSIST_DIR     — ChromaDB dizini (default: ./chroma_db)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# yargi-mcp path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class IndexProgress:
    """Index pipeline ilerlemesi — checkpoint için."""
    keyword: str
    court_type: str
    target_docs: int
    fetched: int = 0
    indexed: int = 0
    failed: int = 0
    chunks_created: int = 0
    last_document_id: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["elapsed_s"] = round(time.time() - self.started_at, 1)
        return d


@dataclass
class IndexResult:
    """Index pipeline sonucu."""
    total_docs_target: int
    total_docs_fetched: int
    total_docs_indexed: int
    total_docs_failed: int
    total_chunks: int
    total_tokens: int
    elapsed_s: float
    embedding_model: str
    chroma_count: int  # ChromaDB'deki toplam kayıt (chunk)
    errors: List[str] = field(default_factory=list)


class BedestenIndexer:
    """
    Bedesten → ChromaDB pipeline.

    Usage:
        indexer = BedestenIndexer()
        result = await indexer.run(
            keywords=["muvazaa tapu iptal", "muris muvazaa"],
            court_types=["YARGITAYKARARI"],
            target_docs=200,
        )
        print(f"{result.total_chunks} chunk indexlendi, {result.elapsed_s:.0f}s")
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        chroma_collection: Optional[str] = None,
    ):
        # Lazy
        self._embedder = None
        self._vector_store = None
        self._bedesten_client = None
        self._chunker = None

        self.batch_size = batch_size or int(os.getenv("INDEXER_BATCH_SIZE", "32"))
        self.chroma_collection = chroma_collection or os.getenv(
            "CHROMA_COLLECTION", "yargi_decisions"
        )

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from semantic_search.embedder import get_embedder
            self._embedder = get_embedder()
            logger.info(
                f"Indexer embedder: {self._embedder.model} "
                f"({self._embedder.dimension}d)"
            )
        return self._embedder

    def _get_vector_store(self):
        if self._vector_store is None:
            from semantic_search.vector_store_chroma import ChromaVectorStore
            embedder = self._get_embedder()
            self._vector_store = ChromaVectorStore(
                dimension=embedder.dimension,
                collection_name=self.chroma_collection,
            )
        return self._vector_store

    def _get_chunker(self):
        if self._chunker is None:
            from .chunker import LegalChunker
            self._chunker = LegalChunker(
                target_tokens=512,
                overlap_tokens=80,
                min_tokens=50,
                max_tokens=700,
            )
        return self._chunker

    async def _get_bedensten_client(self):
        if self._bedesten_client is None:
            # mcp_server_main içindeki global bedesten_client_instance'ı kullan
            import mcp_server_main as mcp
            # bedesten_client_instance module-level singleton
            self._bedesten_client = getattr(mcp, "bedesten_client_instance", None)
            if self._bedesten_client is None:
                # Yoksa kendimiz oluştur
                from bedesten_mcp_module.client import BedestenApiClient
                self._bedesten_client = BedestenApiClient()
                logger.info("Yeni BedestenApiClient oluşturuldu (fallback)")
            else:
                logger.info("mcp_server_main.bedensten_client_instance kullanılıyor (shared)")
        return self._bedesten_client

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def run(
        self,
        keywords: Optional[List[str]] = None,
        court_types: Optional[List[str]] = None,
        target_docs: Optional[int] = None,
        progress_callback=None,
    ) -> IndexResult:
        """
        Tam index pipeline.

        Args:
            keywords: Bedesten search anahtar kelimeleri (her biri ayrı search)
            court_types: Mahkeme tipleri (YARGITAYKARARI, DANISTAYKARAR, vb.)
            target_docs: Toplam hedef belge sayısı
            progress_callback: Optional async callback (progress_dict) -> None

        Returns:
            IndexResult — özet istatistikler
        """
        keywords = keywords or [
            k.strip() for k in os.getenv(
                "INDEXER_KEYWORDS", "muvazaa,tapu,iptal,muris"
            ).split(",") if k.strip()
        ]
        court_types = court_types or [
            c.strip() for c in os.getenv(
                "INDEXER_COURT_TYPES", "YARGITAYKARARI"
            ).split(",") if c.strip()
        ]
        target_docs = target_docs or int(os.getenv("INDEXER_TARGET_DOCS", "200"))

        logger.info(
            f"Indexer başlıyor — target={target_docs} belge, "
            f"keywords={keywords}, courts={court_types}, batch={self.batch_size}"
        )

        t_start = time.time()
        embedder = self._get_embedder()
        vs = self._get_vector_store()

        # 1. Search Bedesten → karar ID'leri topla
        all_decisions = await self._collect_decision_ids(
            keywords=keywords,
            court_types=court_types,
            target=target_docs,
        )
        logger.info(f"Toplam {len(all_decisions)} benzersiz karar ID bulundu")

        # 2. Her kararı çek, chunk'a böl, embed, Chroma'ya yaz
        progress = IndexProgress(
            keyword=",".join(keywords),
            court_type=",".join(court_types),
            target_docs=min(target_docs, len(all_decisions)),
        )

        errors: List[str] = []
        total_chunks = 0
        total_tokens = 0

        # ChromaDB'de zaten var olan ID'leri atla (resume desteği)
        existing_ids = self._get_existing_doc_ids()
        logger.info(f"ChromaDB'de zaten {len(existing_ids)} belge var (atlanacak)")

        for i, decision in enumerate(all_decisions):
            if progress.indexed >= progress.target_docs:
                break
            if decision.documentId in existing_ids:
                progress.fetched += 1
                continue

            try:
                # Fetch full document
                doc_md = await self._fetch_with_retry(decision.documentId)
                if not doc_md or not doc_md.markdown_content:
                    progress.failed += 1
                    errors.append(f"{decision.documentId}: empty content")
                    continue

                progress.fetched += 1

                # Chunk it
                chunker = self._get_chunker()
                metadata = self._build_metadata(decision)
                chunks = chunker.chunk_document(
                    document_id=decision.documentId,
                    text=doc_md.markdown_content,
                    metadata=metadata,
                )

                if not chunks:
                    progress.failed += 1
                    errors.append(f"{decision.documentId}: no chunks (too short?)")
                    continue

                # Batch embed
                chunk_texts = [c.text for c in chunks]
                chunk_ids = [c.chunk_id for c in chunks]
                doc_ids = [decision.documentId] * len(chunks)
                chunk_metas = [c.metadata for c in chunks]

                # NVIDIA passage embedding — chunk by batch
                embeddings = self._embed_batch(chunker_chunks=chunks)

                # Add to ChromaDB
                vs.add_chunks(
                    chunk_ids=chunk_ids,
                    doc_ids=doc_ids,
                    texts=chunk_texts,
                    embeddings=embeddings,
                    metadata=chunk_metas,
                )

                progress.indexed += 1
                progress.chunks_created += len(chunks)
                total_chunks += len(chunks)
                total_tokens += sum(c.token_count for c in chunks)
                progress.last_document_id = decision.documentId
                progress.last_updated = time.time()

                if progress_callback and (i + 1) % 5 == 0:
                    await progress_callback(progress.to_dict())

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"[{i+1}/{len(all_decisions)}] "
                        f"indexed={progress.indexed}/{progress.target_docs}, "
                        f"chunks={progress.chunks_created}, "
                        f"failed={progress.failed}, "
                        f"elapsed={time.time()-t_start:.0f}s"
                    )

            except Exception as e:
                progress.failed += 1
                errors.append(f"{decision.documentId}: {str(e)[:100]}")
                logger.warning(f"Index hatası {decision.documentId}: {e}")
                continue

        elapsed = time.time() - t_start
        result = IndexResult(
            total_docs_target=progress.target_docs,
            total_docs_fetched=progress.fetched,
            total_docs_indexed=progress.indexed,
            total_docs_failed=progress.failed,
            total_chunks=total_chunks,
            total_tokens=total_tokens,
            elapsed_s=elapsed,
            embedding_model=embedder.model,
            chroma_count=vs.size(),
            errors=errors[:20],  # ilk 20 hata
        )

        logger.info(
            f"Index bitti — {result.total_docs_indexed}/{result.total_docs_target} belge, "
            f"{result.total_chunks} chunk, {result.total_tokens} token, "
            f"{result.elapsed_s:.0f}s, ChromaDB={result.chroma_count} kayıt"
        )

        # Progress dosyaya yaz (checkpoint)
        self._write_checkpoint(progress, result)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _collect_decision_ids(
        self,
        keywords: List[str],
        court_types: List[str],
        target: int,
    ) -> List[Any]:
        """Bedesten search ile yeterli sayıda karar ID topla."""
        import mcp_server_main as mcp
        from bedesten_mcp_module.models import BedestenSearchRequest, BedestenSearchData

        client = await self._get_bedensten_client()
        seen_ids = set()
        all_decisions = []

        # Her keyword × court_type kombinasyonu için search
        for kw in keywords:
            for court in court_types:
                if len(all_decisions) >= target * 2:  # 2x toplama payı
                    break

                per_call = min(50, max(20, target // max(len(keywords), 1)))
                try:
                    req = BedestenSearchRequest(
                        data=BedestenSearchData(
                            phrase=kw,
                            itemTypeList=[court],
                            pageSize=per_call,
                            pageNumber=1,
                        )
                    )
                    resp = await client.search_documents(req)
                    if resp.data and resp.data.emsalKararList:
                        for d in resp.data.emsalKararList:
                            if d.documentId not in seen_ids:
                                seen_ids.add(d.documentId)
                                all_decisions.append(d)
                        logger.info(
                            f"Search '{kw}' / {court} → +{len(resp.data.emsalKararList)} "
                            f"(toplam {len(all_decisions)})"
                        )
                except Exception as e:
                    logger.warning(f"Search hatası '{kw}'/{court}: {e}")

                # Bedesten rate-limit için bekle
                await asyncio.sleep(1.0)

        return all_decisions

    async def _fetch_with_retry(self, doc_id: str, max_retries: int = 3):
        """Bedesten get_document_as_markdown — rate-limit'e karşı retry."""
        from bedesten_mcp_module.client import BedestenRateLimited
        client = await self._get_bedensten_client()

        for attempt in range(max_retries):
            try:
                return await client.get_document_as_markdown(doc_id)
            except BedestenRateLimited as e:
                if attempt < max_retries - 1:
                    wait = min(float(e.retry_after) + 1.0, 60.0)
                    logger.info(
                        f"Doc {doc_id} rate-limited (attempt {attempt+1}/{max_retries}), "
                        f"waiting {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
            except Exception as e:
                # Network/parse errors — retry yapma
                logger.debug(f"Doc {doc_id} fetch hatası: {e}")
                raise

    def _build_metadata(self, decision: Any) -> Dict[str, Any]:
        """Bedesten decision objesinden metadata çıkar."""
        return {
            "document_id": decision.documentId,
            "birim_adi": getattr(decision, "birimAdi", "") or "",
            "esas_no": getattr(decision, "esasNo", "") or "",
            "karar_no": getattr(decision, "kararNo", "") or "",
            "karar_tarihi": getattr(decision, "kararTarihiStr", "") or "",
            "court_type": getattr(decision, "itemType", None).name
                          if getattr(decision, "itemType", None) else "",
        }

    def _embed_batch(self, chunker_chunks: List[Any]) -> np.ndarray:
        """
        NVIDIA nv-embed-v1 ile batch embedding.
        batch_size'ı aşarsa parça parça embed'le.
        """
        embedder = self._get_embedder()
        texts = [c.text for c in chunker_chunks]
        titles = [c.metadata.get("birim_adi", "") or "none" for c in chunker_chunks]

        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_titles = titles[i:i + self.batch_size]

            # NVIDIA nv-embed-v1 asimetrik: passage input_type
            embs = embedder.encode_documents(batch_texts, titles=batch_titles)
            if isinstance(embs, np.ndarray):
                all_embs.append(embs)
            else:
                all_embs.append(np.array(embs))

            # NVIDIA rate-limit için mini bekle
            if i + self.batch_size < len(texts):
                time.sleep(0.3)

        return np.vstack(all_embs) if all_embs else np.array([])

    def _get_existing_doc_ids(self) -> set:
        """ChromaDB'de var olan tüm document_id'leri getir (resume desteği)."""
        try:
            vs = self._get_vector_store()
            # Tüm metadataları çek
            res = vs.collection.get(include=["metadatas"])
            ids = set()
            for meta in res.get("metadatas", []):
                did = meta.get("document_id")
                if did:
                    ids.add(did)
            return ids
        except Exception as e:
            logger.warning(f"Existing IDs alınamadı: {e}")
            return set()

    def _write_checkpoint(self, progress: IndexProgress, result: IndexResult):
        """Index sonucunu JSON checkpoint'a yaz."""
        try:
            ckpt_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "last_index.json")
            payload = {
                "progress": progress.to_dict(),
                "result": {
                    "total_docs_target": result.total_docs_target,
                    "total_docs_indexed": result.total_docs_indexed,
                    "total_chunks": result.total_chunks,
                    "total_tokens": result.total_tokens,
                    "elapsed_s": result.elapsed_s,
                    "embedding_model": result.embedding_model,
                    "chroma_count": result.chroma_count,
                    "errors_count": len(result.errors),
                    "timestamp": time.time(),
                },
            }
            with open(ckpt_path, "w") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f"Checkpoint yazıldı: {ckpt_path}")
        except Exception as e:
            logger.warning(f"Checkpoint yazılamadı: {e}")
