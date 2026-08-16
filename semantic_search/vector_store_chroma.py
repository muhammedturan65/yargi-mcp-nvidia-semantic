"""
ChromaDB-backed vector store — kalıcı, sub-second retrieval.

Bu modül, `vector_store.py`'daki in-memory VectorStore ile aynı API'yi sağlar
(drop-in replacement), ancak embedding'leri disk'e yazar. Process restart'ında
kayıp olmaz; ilk sorgudan sonraki sorgular 50ms altında döner.

Tasarım kararları:
  - Her belge birden fazla chunk içerebilir (chunk = tek bir embedding).
    `add_documents()` "document-level" add yapar; `add_chunks()` chunk-level.
    Chunk ID formatı: `{doc_id}__c{idx}` — böylece document_id'ye göre filtre
    yapılabiliyor ( Chroma `where` clause).
  - Cosine similarity: Chroma's default `cosine` kullanılır. NVIDIA nv-embed-v1
    normalize edilmiş embedding döndürür, ama Chroma yine de normalize edip
    güvenli skor üretir.
  - Collection isimleri: court-type bazında ayrılabilir. Default `yargi_decisions`.
  - Persistent path: env `CHROMA_PERSIST_DIR` (default: `./chroma_db`).

Env vars:
  CHROMA_PERSIST_DIR       — kalıcı dizin (default: ./chroma_db)
  CHROMA_COLLECTION        — collection adı (default: yargi_decisions)
  CHROMA_DISTANCE          — distance metric: cosine|l2|ip (default: cosine)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ChromaDB'yi lazy import et — bağımlılık sadece bu modül kullanıldığında gerekli.
_chroma_client = None
_chroma_collections: Dict[str, Any] = {}


def _get_chroma_client():
    """Singleton ChromaDB client. İlk çağrıda persistent client açar."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    import chromadb
    from chromadb.config import Settings

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    os.makedirs(persist_dir, exist_ok=True)

    _chroma_client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    logger.info(f"ChromaDB persistent client açıldı: {persist_dir}")
    return _chroma_client


def _get_collection(name: Optional[str] = None, dimension: Optional[int] = None):
    """Get-or-create a collection. Dimension sadece metadata'ya yazılır (bilgi)."""
    client = _get_chroma_client()
    coll_name = name or os.getenv("CHROMA_COLLECTION", "yargi_decisions")
    distance = os.getenv("CHROMA_DISTANCE", "cosine")

    if coll_name in _chroma_collections:
        return _chroma_collections[coll_name]

    try:
        coll = client.get_collection(name=coll_name)
        logger.info(f"ChromaDB collection mevcut: {coll_name} ({coll.count()} kayıt)")
    except Exception:
        coll = client.create_collection(
            name=coll_name,
            metadata={
                "hnsw:space": distance,
                "dimension": str(dimension or 0),
            },
        )
        logger.info(f"ChromaDB collection oluşturuldu: {coll_name} (distance={distance})")

    _chroma_collections[coll_name] = coll
    return coll


class ChromaVectorStore:
    """
    Kalıcı vector store — ChromaDB backend.

    `VectorStore` (in-memory) ile API-uyumlu: add_documents / search / size /
    clear / get_stats metodları aynı imza ve dönüş tipleriyle sağlanır.
    Ek olarak `add_chunks()` ile chunk-level indexing yapılabilir.

    Kullanım:
        store = ChromaVectorStore(dimension=4096)
        store.add_documents(ids=[...], texts=[...], embeddings=arr, metadata=[...])
        results = store.search(query_emb, top_k=5)
    """

    def __init__(
        self,
        dimension: int = 768,
        collection_name: Optional[str] = None,
    ):
        self.dimension = dimension
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "yargi_decisions"
        )
        self.collection = _get_collection(self.collection_name, dimension=dimension)
        logger.info(
            f"ChromaVectorStore init — collection='{self.collection_name}', "
            f"dimension={dimension}, mevcut kayıt={self.collection.count()}"
        )

    # ------------------------------------------------------------------
    # Document-level API (vector_store.py ile uyumlu)
    # ------------------------------------------------------------------

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Add document-level embeddings to ChromaDB.

        Aynı ID ile tekrar eklersek ChromaDB upsert yapar (üzerine yazar).
        """
        if len(ids) != len(texts) or len(ids) != embeddings.shape[0]:
            raise ValueError("Mismatched lengths for ids, texts, and embeddings")
        if metadata and len(metadata) != len(ids):
            raise ValueError("Metadata length doesn't match document count")

        # ChromaDB embeddings'i list-of-list olarak bekler
        emb_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        metadatas = metadata if metadata else [{}] * len(ids)

        # ChromaDB metadata değerleri primitive tipte olmalı (str/int/float/bool/None)
        clean_metas = [_sanitize_chroma_meta(m) for m in metadatas]

        # Upsert — aynı ID varsa üzerine yazar (idempotent re-index)
        self.collection.upsert(
            ids=ids,
            embeddings=emb_list,
            documents=texts,
            metadatas=clean_metas,
        )
        logger.info(f"ChromaDB'ye {len(ids)} belge upsert edildi. Toplam: {self.collection.count()}")
        return len(ids)

    # ------------------------------------------------------------------
    # Chunk-level API (chunker ile birlikte kullanılır)
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunk_ids: List[str],
        doc_ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> int:
        """
        Chunk-level index. Her chunk ayrı bir embedding alır.

        `metadata` içinde chunk_index, total_chunks, document_id olmalıdır.
        Bu sayede retrieve sonrası aynı document_id'ye ait chunk'lar
        gruplanabilir (dedup).
        """
        if not (len(chunk_ids) == len(doc_ids) == len(texts) == embeddings.shape[0]):
            raise ValueError("chunk_ids, doc_ids, texts, embeddings boyutları eşit olmalı")

        emb_list = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
        clean_metas = []
        for did, m in zip(doc_ids, metadata):
            cm = _sanitize_chroma_meta(m)
            cm["document_id"] = did  # filter için guarantee
            clean_metas.append(cm)

        self.collection.upsert(
            ids=chunk_ids,
            embeddings=emb_list,
            documents=texts,
            metadatas=clean_metas,
        )
        logger.info(
            f"ChromaDB'ye {len(chunk_ids)} chunk upsert edildi "
            f"({len(set(doc_ids))} belge). Toplam: {self.collection.count()}"
        )
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
        where: Optional[Dict] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Cosine similarity search.

        Returns:
            List of (Document-like, score) tuples. Document objesi
            `id`, `text`, `metadata` alanları içerir (numpy embedding yok).
        """
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection boş")
            return []

        q = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        if hasattr(q, "__len__") and len(q) > 0 and isinstance(q[0], list):
            q = q[0]  # 2D ise 1D'ye indir

        n = max(top_k, 10)  # Chroma min 10 döner; biz slicing yaparız
        n = min(n, self.collection.count())

        query_args = {
            "query_embeddings": [q],
            "n_results": n,
        }
        if where:
            query_args["where"] = where

        try:
            res = self.collection.query(**query_args)
        except Exception as e:
            logger.error(f"ChromaDB query hatası: {e}")
            return []

        # Sonuçları unpack et — Chroma her zaman list-of-list döner
        ids_list = res.get("ids", [[]])[0]
        docs_list = res.get("documents", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        dists_list = res.get("distances", [[]])[0]

        # Chroma cosine distance döner (0 = identical, 2 = opposite).
        # similarity = 1 - distance
        results: List[Tuple[Any, float]] = []
        for cid, doc_text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            sim = 1.0 - float(dist)
            if threshold is not None and sim < threshold:
                continue
            doc = _ChromaDocument(
                id=cid,
                text=doc_text or "",
                metadata=meta or {},
            )
            results.append((doc, sim))

        # Skora göre azalan sırala ve top_k'ya kes
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        logger.info(f"ChromaDB search: {len(results)} sonuç (top_k={top_k})")
        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def search_with_dedup(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Chunk-level arama yapıp, document bazında dedup yapar.
        Her belge için en yüksek skorlu chunk'ı döndürür.

        Bu metod chunk'lı index kullanıldığında kaliteyi artırır.

        Threshold NOT applied during raw search — sadece final top_k
        sonucuna uygulanır. Bu sayede düşük skorlu ama ilgili chunk'lar
        kaçırılmaz (chunk bazında skor, document bazında daha düşük olabilir).
        """
        # 3-5x fazla chunk getir (threshold yok), sonra doc-level dedup yap
        raw_k = min(top_k * 5, 50)
        raw_results = self.search(query_embedding, top_k=raw_k, threshold=None)

        seen_docs: Dict[str, Tuple[Any, float]] = {}
        for doc, score in raw_results:
            doc_id = doc.metadata.get("document_id", doc.id)
            if doc_id not in seen_docs or score > seen_docs[doc_id][1]:
                # İlk chunk'ı (en yüksek skorluyu) tut, ama metadata'yı zenginleştir
                enriched = _ChromaDocument(
                    id=doc_id,
                    text=doc.text,
                    metadata={**doc.metadata, "chunk_id": doc.id, "best_chunk_score": score},
                )
                seen_docs[doc_id] = (enriched, score)

        final = list(seen_docs.values())
        # Threshold'u sadece final listeye uygula
        if threshold is not None:
            final = [(d, s) for d, s in final if s >= threshold]
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:top_k]

    def clear(self):
        """Tüm collection'ı sil ve yeniden oluştur."""
        client = _get_chroma_client()
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"ChromaDB collection silindi: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Collection silinemedi (yok?): {e}")
        _chroma_collections.pop(self.collection_name, None)
        self.collection = _get_collection(self.collection_name, dimension=self.dimension)

    def size(self) -> int:
        """Number of records (chunk sayısı)."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_by_id(self, doc_id: str) -> Optional[Any]:
        """ID'ye göre tek kayıt getir."""
        try:
            res = self.collection.get(ids=[doc_id])
            if not res["ids"]:
                return None
            return _ChromaDocument(
                id=res["ids"][0],
                text=res["documents"][0] if res["documents"] else "",
                metadata=res["metadatas"][0] if res["metadatas"] else {},
            )
        except Exception as e:
            logger.warning(f"get_by_id hatası: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Statistics about the vector store."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0

        # ChromaDB memory estimate yaklaşık — gerçek disk kullanımı farklıdır
        # nv-embed-v1: 4096 float32 = 16KB per chunk
        est_memory_mb = (count * self.dimension * 4) / (1024 * 1024)

        return {
            "num_documents": count,
            "dimension": self.dimension,
            "index_built": count > 0,
            "memory_usage_mb": round(est_memory_mb, 2),
            "backend": "chromadb",
            "collection": self.collection_name,
            "persist_dir": os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        }

    def list_documents_by_metadata(self, where: Dict, limit: int = 100) -> List[Dict]:
        """Metadata'ya göre sorgula (örn: {'court_type': 'YARGITAYKARARI'})."""
        try:
            res = self.collection.get(where=where, limit=limit)
            return [
                {
                    "id": cid,
                    "text": doc,
                    "metadata": meta,
                }
                for cid, doc, meta in zip(
                    res["ids"], res["documents"], res["metadatas"]
                )
            ]
        except Exception as e:
            logger.warning(f"list_documents_by_metadata hatası: {e}")
            return []


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


class _ChromaDocument:
    """
    ChromaDB sonuçlarını vector_store.Document formatına uyarlar.

    Embedding yok (ChromaDB gerek yok — ihtiyaç olursa ayrı çekilir).
    `to_dict()` metodu mevcut Document ile uyumlu.
    """

    def __init__(self, id: str, text: str, metadata: Dict[str, Any]):
        self.id = id
        self.text = text
        self.metadata = metadata
        # Embedding field for compatibility with Document dataclass
        self.embedding = None

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}


def _sanitize_chroma_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB metadata değerleri sadece primitive tipte olabilir.
    List/dict/None değerlerini string'e çevirir.
    """
    clean = {}
    for k, v in meta.items():
        if v is None:
            continue  # None'ı atla
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, dict)):
            clean[k] = str(v)
        else:
            clean[k] = str(v)
    return clean
