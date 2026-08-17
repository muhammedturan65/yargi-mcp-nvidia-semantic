"""
RAG engine — yargi-mcp semantik arama + multi-provider LLM + answer cache + query cache.

Kullanıcı sorusu → embedding → vector store'da top-K arama →
context construction → LLM ile cevap üretimi →
atıflı yanıt.

v1.4.0:
    - Query embedding cache (RAG_QUERY_CACHE=true, default)
      Aynı soru tekrar sorulduğunda NVIDIA query embedding çağrısı yapılmaz.
      İki katmanlı: in-memory LRU (0 ms) + ChromaDB persistent (5 ms).
      v1.3.0'da her sorguda ~1 s NVIDIA'ya gidiyordu, artık sadece ilk seferde.
    - RAGResponse'a query_cache_hit + query_cache_source alanları eklendi.

v1.3.0:
    - Multi-provider LLM (LLM_PROVIDER=nvidia|groq|openai|ollama)
    - Semantik answer cache (RAG_ANSWER_CACHE=true, default)
      Aynı/benzer soru tekrar sorulduğunda LLM çağrısı yapılmaz.
    - LLMResponse.from_cache alanı cache hit olduğunu belirtir.

v1.2.0:
    - ChromaDB kalıcı vector store (restart'ta kayıp yok)
    - Token-aware chunking (512 token, 80 overlap)
    - Bedesten tam metin çekme (preview yerine)

İki backend modu var:
    - "chroma" (default): ChromaDB kalıcı vector store. İlk çağrıda Bedesten
      üzerinden ~200 belge çekilir, 512-token chunk'lara bölünür, NVIDIA
      nv-embed-v1 ile embed'lenir ve diske yazılır. Sonraki sorgular 50ms
      altında döner — process restart'ında veri kaybı yok.
    - "memory" (legacy): In-memory VectorStore. Her sorguda search_bedesten_semantic
      çağrılır, ~2 dk sürer. v1.1.0 davranışının korunduğu fallback modu.

Kullanım:
    rag = LegalQARAG()                       # chroma backend (default)
    await rag.load_corpora()                 # ilk seferlikte ~5-10 dk
    response = await rag.ask("Mirasçı hangi davayı açar?")  # sub-second retrieval

    # Hızlı LLM (önerilen):
    os.environ["LLM_PROVIDER"] = "groq"
    os.environ["GROQ_API_KEY"] = "gq_..."
    rag = LegalQARAG()
    response = await rag.ask("...")  # ~3s yerine ~60s

    # Cache kapatma:
    os.environ["RAG_ANSWER_CACHE"] = "false"
    os.environ["RAG_QUERY_CACHE"] = "false"
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# yargi-mcp path'ini ekle (üst dizin)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@dataclass
class RAGContext:
    """Retrieval sonucu — LLM'e girecek context."""
    question: str
    decisions: List[Dict] = field(default_factory=list)  # formatted_results formatı
    total_found: int = 0
    search_time_ms: float = 0.0
    query_embedding: Optional[object] = None  # v1.3.0+: reuse for cache lookup
    query_cache_hit: bool = False              # v1.4.0+: query embedding cache hit mi
    query_cache_source: str = "miss"           # v1.4.0+: "lru" | "persistent" | "miss"


@dataclass
class RAGResponse:
    """Tam RAG yanıtı — cevap + atıflar + metadata."""
    question: str
    answer: str
    citations: List[Dict]          # formatted_results'un aynısı
    context_decision_count: int
    embedding_model: str
    llm_model: str
    llm_usage: Dict[str, int]      # prompt/completion/total tokens
    total_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    # v1.3.0+ — answer cache metadata
    from_cache: bool = False
    cache_score: float = 0.0       # cache hit ise cosine score
    llm_provider: str = ""
    # v1.4.0+ — query embedding cache metadata
    query_cache_hit: bool = False
    query_cache_source: str = "miss"   # "lru" | "persistent" | "miss"


class LegalQARAG:
    """
    Hukuki QA RAG pipeline.

    Args:
        backend: "chroma" (default, kalıcı) veya "memory" (legacy, in-memory)
        n_decisions_per_query: Kaç karar çekilecek (memory mode için)
        top_k_retrieval: LLM'e kaç karar feed'lenecek
        llm_temperature: Hukuki: düşük yaratıcılık (0.2)
        llm_max_tokens: Maksimum cevap token sayısı
        chroma_collection: ChromaDB collection adı (chroma backend için)
        llm_provider: v1.3.0+ — "nvidia"|"groq"|"openai"|"ollama" (default: env LLM_PROVIDER)
        enable_answer_cache: v1.3.0+ — semantik answer cache (default: env RAG_ANSWER_CACHE)
        enable_query_cache: v1.4.0+ — query embedding cache (default: env RAG_QUERY_CACHE)
        enable_section_aware: v1.5.0+ — section-aware reranking (default: env RAG_SECTION_AWARE)

    Usage:
        rag = LegalQARAG()                       # chroma backend (default)
        await rag.load_corpora()                 # ilk seferlikte index
        response = await rag.ask("Mirasçı hangi davayı açar?")

        # Hızlı LLM (önerilen v1.3.0+):
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "gq_..."
        rag = LegalQARAG()
        response = await rag.ask("...")  # ~3s yerine ~60s
    """

    def __init__(
        self,
        backend: str = "chroma",
        n_decisions_per_query: int = 30,
        top_k_retrieval: int = 5,
        llm_temperature: float = 0.2,
        llm_max_tokens: int = 1500,
        chroma_collection: Optional[str] = None,
        llm_provider: Optional[str] = None,
        enable_answer_cache: Optional[bool] = None,
        enable_query_cache: Optional[bool] = None,
        enable_section_aware: Optional[bool] = None,
    ):
        if backend not in ("chroma", "memory"):
            raise ValueError(f"backend 'chroma' veya 'memory' olmalı, got: {backend}")
        self.backend = backend
        self.n_decisions_per_query = n_decisions_per_query
        self.top_k_retrieval = top_k_retrieval
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.chroma_collection = chroma_collection or os.getenv(
            "CHROMA_COLLECTION", "yargi_decisions"
        )
        self.llm_provider = llm_provider  # None → LLMClient env'den çözer

        # v1.3.0+ — Answer cache (ChromaDB'de ayrı collection)
        if enable_answer_cache is not None:
            os.environ["RAG_ANSWER_CACHE"] = "true" if enable_answer_cache else "false"
        self._answer_cache = None  # lazy init

        # v1.4.0+ — Query embedding cache (LRU + ChromaDB persistent)
        if enable_query_cache is not None:
            os.environ["RAG_QUERY_CACHE"] = "true" if enable_query_cache else "false"
        self._query_cache = None  # lazy init

        # v1.5.0+ — Section-aware retrieval scoring
        if enable_section_aware is not None:
            os.environ["RAG_SECTION_AWARE"] = "true" if enable_section_aware else "false"
        self.section_aware = os.getenv("RAG_SECTION_AWARE", "true").lower() in (
            "true", "1", "yes", "on",
        )

        # Lazy-loaded components
        self._embedder = None
        self._vector_store = None
        self._llm_client = None
        self._mcp_module = None
        self._indexer = None
        self._is_corpora_loaded = False

    # ---------- Lazy initialization ----------

    def _get_embedder(self):
        if self._embedder is None:
            from semantic_search.embedder import get_embedder
            self._embedder = get_embedder()
            logger.info(f"Embedder yüklendi: {self._embedder.model} ({self._embedder.dimension}d)")
        return self._embedder

    def _get_vector_store(self):
        if self._vector_store is None:
            embedder = self._get_embedder()
            if self.backend == "chroma":
                from semantic_search.vector_store_chroma import ChromaVectorStore
                self._vector_store = ChromaVectorStore(
                    dimension=embedder.dimension,
                    collection_name=self.chroma_collection,
                )
                logger.info(
                    f"ChromaVectorStore oluşturuldu (dimension={embedder.dimension}, "
                    f"collection={self.chroma_collection}, mevcut={self._vector_store.size()} kayıt)"
                )
            else:
                from semantic_search.vector_store import VectorStore
                self._vector_store = VectorStore(dimension=embedder.dimension)
                logger.info(f"In-memory VectorStore oluşturuldu (dimension={embedder.dimension})")
        return self._vector_store

    def _get_llm_client(self):
        """v1.3.0+: Multi-provider LLM client. Default NVIDIA (backward compat)."""
        if self._llm_client is None:
            from .llm_client import LLMClient
            kwargs = {
                "temperature": self.llm_temperature,
                "max_tokens": self.llm_max_tokens,
            }
            if self.llm_provider:
                kwargs["provider"] = self.llm_provider
            self._llm_client = LLMClient(**kwargs)
            logger.info(
                f"LLM client yüklendi: provider={self._llm_client.provider}, "
                f"model={self._llm_client.model}"
            )
        return self._llm_client

    def _get_answer_cache(self):
        """v1.3.0+: Lazy-init answer cache (ChromaDB'de ayrı collection)."""
        if self._answer_cache is None:
            from .answer_cache import AnswerCache
            embedder = self._get_embedder()
            self._answer_cache = AnswerCache(dimension=embedder.dimension)
        return self._answer_cache

    def _get_query_cache(self):
        """v1.4.0+: Lazy-init query embedding cache (LRU + ChromaDB persistent)."""
        if self._query_cache is None:
            from .query_cache import QueryEmbeddingCache
            embedder = self._get_embedder()
            self._query_cache = QueryEmbeddingCache(dimension=embedder.dimension)
        return self._query_cache

    def _get_mcp_module(self):
        """mcp_server_main modülünü lazy yükle (search_bedesten_semantic için)."""
        if self._mcp_module is None:
            import mcp_server_main
            self._mcp_module = mcp_server_main
            logger.info("mcp_server_main yüklendi (Bedesten semantic search için)")
        return self._mcp_module

    def _get_indexer(self):
        """ChromaDB indexer (lazy)."""
        if self._indexer is None:
            from .indexer import BedestenIndexer
            self._indexer = BedestenIndexer(chroma_collection=self.chroma_collection)
            logger.info("BedestenIndexer oluşturuldu")
        return self._indexer

    # ---------- Public API ----------

    @property
    def is_corpora_loaded(self) -> bool:
        """Vector store'da en az 1 karar var mı?

        Chroma backend'de bu, ChromaDB'de kayıt varsa True döner — process
        restart'ında bile. Memory backend'de sadece runtime'da yüklenmişse True.
        """
        if self.backend == "chroma":
            # ChromaDB'yi lazy init — kaç kayıt var?
            vs = self._get_vector_store()
            return vs.size() > 0
        else:
            if self._vector_store is None:
                return False
            return self._vector_store.size() > 0

    async def load_corpora(
        self,
        initial_keyword: str = "muvazaa tapu iptal",
        semantic_query: str = "Mirasçının muvazaalı satış işlemine karşı tapu iptali ve tescil davası açması",
        court_types: Optional[List[str]] = None,
        target_docs: Optional[int] = None,
    ) -> Dict:
        """
        Bedesten API'den karar çekip vector store'a yükle.

        Chroma backend (v1.2.0):
            - BedestenIndexer ile TAM METİN çekilir, chunk'lanır, NVIDIA ile
              embed'lenir ve ChromaDB'ye yazılır.
            - İlk çağrı 5-10 dk sürebilir (200 belge × 3.5s Bedesten rate-limit).
            - ChromaDB'de zaten var olan belgeler atlanır (resume desteği).
            - Process restart'ında tekrar çağrılmasına gerek yok.

        Memory backend (legacy v1.1.0):
            - search_bedesten_semantic tool'unu çağırır, ~2 dk sürer.
            - Process restart'ında kaybolur.

        Args:
            initial_keyword: Bedesten search anahtar kelimesi
            semantic_query: Anlamsal sorgu (sadece memory backend için kullanılır)
            court_types: Mahkeme tipleri (default: ["YARGITAYKARARI"])
            target_docs: Chroma backend için hedef belge sayısı (default: env veya 200)

        Returns:
            Dict — backend'e göre:
              chroma: {status, indexed, chunks, elapsed_s, ...}
              memory: search_bedesten_semantic'in döndüğü dict
        """
        if court_types is None:
            court_types = ["YARGITAYKARARI"]

        # ---------- Chroma backend ----------
        if self.backend == "chroma":
            indexer = self._get_indexer()
            # Keyword'ü virgülle ayrılmış listeye çevür (indexer multi-keyword destekler)
            keywords = [k.strip() for k in initial_keyword.split(",") if k.strip()]
            if not keywords:
                keywords = [initial_keyword]

            target = target_docs or int(os.getenv("INDEXER_TARGET_DOCS", "200"))

            logger.info(
                f"ChromaDB index başlıyor: keywords={keywords}, "
                f"courts={court_types}, target={target}"
            )

            result = await indexer.run(
                keywords=keywords,
                court_types=court_types,
                target_docs=target,
            )

            self._is_corpora_loaded = True
            return {
                "status": "success",
                "backend": "chroma",
                "indexed_docs": result.total_docs_indexed,
                "total_chunks": result.total_chunks,
                "total_tokens": result.total_tokens,
                "elapsed_s": round(result.elapsed_s, 1),
                "embedding_model": result.embedding_model,
                "chroma_count": result.chroma_count,
                "failed_docs": result.total_docs_failed,
                "errors": result.errors[:5],
                "stats": {
                    "documents_in_store": result.chroma_count,
                    "failed_fetches": result.total_docs_failed,
                },
            }

        # ---------- Memory backend (legacy) ----------
        mcp = self._get_mcp_module()
        if not getattr(mcp, "SEMANTIC_SEARCH_AVAILABLE", False):
            raise RuntimeError(
                "yargi-mcp semantik arama aktif değil. NVIDIA env vars'ları kontrol edin."
            )

        logger.info(f"Corpora yükleniyor (memory mode): keyword='{initial_keyword}'")
        result = await mcp.search_bedesten_semantic(
            initial_keyword=initial_keyword,
            query=semantic_query,
            court_types=court_types,
            top_k=self.n_decisions_per_query,
        )

        if result.get("status") != "success":
            raise RuntimeError(f"Bedesten search hatası: {result.get('message', result)}")

        # search_bedesten_semantic kendi vector_store'una ekledi.
        self._vector_store = getattr(mcp, "_qa_rag_vector_store", None) or self._vector_store

        if self._vector_store.size() == 0:
            await self._populate_store_from_results(result)

        self._is_corpora_loaded = True
        logger.info(
            f"Corpora yüklendi: {result.get('total_documents_processed')} karar, "
            f"{self._vector_store.size()} vector store'da"
        )
        return result

    async def _populate_store_from_results(self, search_result: Dict):
        """
        search_bedesten_semantic'in döndürdüğü results listesinden kendi
        vector_store'umuzu populate et. mcp_server_main'in local vector_store
        değişkenine erişemediğimiz için bu sarma yöntemi kullanıyoruz.

        İleride refactor: mcp_server_main vector_store'u module-level singleton
        yapılacak.
        """
        from semantic_search.vector_store import VectorStore

        embedder = self._get_embedder()
        vs = self._get_vector_store()

        # mcp_server_main'in populate ettiği store'u doğrudan kullanmak için
        # module-level vector_store değişkenini ara
        import mcp_server_main as mcp

        # Çeşitli isimlerle dene
        for attr_name in ["vector_store", "_vector_store", "global_vector_store"]:
            store = getattr(mcp, attr_name, None)
            if store is not None and hasattr(store, "documents") and store.size() > 0:
                logger.info(f"mcp_server_main.{attr_name} bulundu — paylaşılıyor")
                self._vector_store = store
                return

        # Bulunamazsa, results listesinden yeniden embed et
        logger.warning(
            "mcp_server_main vector_store bulunamadı — results'tan yeniden embed ediliyor"
        )
        results = search_result.get("results", [])
        if not results:
            return

        # Her sonuç için: preview'ı al, tekrar embed et, store'a ekle
        texts = []
        metadatas = []
        ids = []
        for r in results:
            # preview'da "..." var, full text yok — metadata yeterli
            text = r.get("preview", "").rstrip(".").rstrip()
            texts.append(text)
            metadatas.append(r.get("metadata", {}))
            ids.append(r["document_id"])

        if texts:
            embeddings = embedder.encode_documents(texts)
            vs.add_documents(
                ids=ids,
                texts=texts,
                embeddings=embeddings,
                metadata=metadatas,
            )

    async def retrieve(self, question: str, top_k: Optional[int] = None) -> RAGContext:
        """
        Soru için en alakalı K kararı getir.

        v1.4.0+ flow:
            1. Query cache lookup (LRU → ChromaDB persistent)
               - HIT  → cached embedding kullan, NVIDIA'ya GİTME
               - MISS → NVIDIA encode_query çağır, cache'e yaz
            2. ChromaDB search_with_dedup (chunk-level → doc dedup)
            3. RAGContext (query_cache_hit + query_cache_source ile)

        Chroma backend'de search_with_dedup kullanılır — chunk-level arama
        yapılıp document bazında dedup edilir. Bu, hem hızlı (50ms) hem de
        kaliteli (en alakalı chunk'a göre sıralama) sonuç verir.

        Returns:
            RAGContext — decisions listesi formatted_results formatında
        """
        t0 = time.time()

        if not self.is_corpora_loaded:
            raise RuntimeError(
                "Corpora yüklenmemiş. Önce rag.load_corpora() çağırın."
            )

        embedder = self._get_embedder()
        vs = self._vector_store

        # v1.4.0+ — Query cache lookup (LRU → ChromaDB persistent → NVIDIA)
        query_cache = self._get_query_cache()
        cache_hit = query_cache.lookup(question) if query_cache.enabled else None

        if cache_hit is not None:
            query_emb = cache_hit.embedding
            cache_source = cache_hit.source  # "lru" or "persistent"
            logger.info(
                f"Query cache HIT ({cache_source}) — NVIDIA çağrısı atlandı"
            )
        else:
            # Cache MISS — NVIDIA API çağrısı yap
            t_nvidia_start = time.time()
            query_emb = embedder.encode_query(question, task="legal question answering")
            nvidia_ms = (time.time() - t_nvidia_start) * 1000
            if isinstance(query_emb, np.ndarray) and query_emb.ndim == 2:
                query_emb = query_emb[0]
            logger.info(
                f"Query cache MISS — NVIDIA encode_query {nvidia_ms:.0f}ms"
            )
            # Cache'e yaz (LRU + persistent)
            if query_cache.enabled:
                query_cache.store(question, query_emb)
            cache_source = "miss"

        # Search
        k = top_k or self.top_k_retrieval

        # Chroma backend: chunk-level search + doc dedup
        # threshold 0.15 — Chroma cosine skorları NVIDIA passage/query arası
        # genelde 0.3-0.55 arası, ama kısa sorgularda 0.15'e düşebilir.
        if self.backend == "chroma" and hasattr(vs, "search_with_dedup"):
            # v1.5.0+: 3x daha fazla chunk çek, sonra section-aware rerank yapacağız
            raw_k = k * 3 if self.section_aware else k
            results = vs.search_with_dedup(query_emb, top_k=raw_k, threshold=0.15)
        else:
            results = vs.search(query_emb, top_k=k)

        # v1.5.0+ — Section-aware reranking
        # Hukuki kararlarda GEREKÇE ve HÜKÜM bölümleri en yüksek değere sahip.
        # Bir sorguyla en alakalı chunk GEREKÇE'de ise, skoru boost'lanır.
        if self.section_aware and self.backend == "chroma":
            results = self._rerank_by_section(results)

        # Top-k'ya kırp
        results = results[:k]

        decisions = []
        for doc, score in results:
            title_parts = []
            md = doc.metadata
            # document_id baz al: chunk-level search'te doc.id = chunk_id olabilir
            doc_id = md.get("document_id", doc.id)
            if md.get("birim_adi"):
                title_parts.append(md["birim_adi"])
            if md.get("esas_no"):
                title_parts.append(f"Esas: {md['esas_no']}")
            if md.get("karar_no"):
                title_parts.append(f"Karar: {md['karar_no']}")
            if md.get("karar_tarihi"):
                title_parts.append(f"Tarih: {md['karar_tarihi']}")

            # chunk-level metadata'da section bilgisi varsa, zenginleştir
            section = md.get("section")
            if section and section != "body":
                title_parts.append(f"[{section}]")

            decisions.append({
                "document_id": doc_id,
                "chunk_id": doc.id if doc.id != doc_id else None,
                "title": " - ".join(title_parts) if title_parts else f"Document {doc_id}",
                "similarity_score": float(score),
                "preview": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                "text": doc.text,
                "metadata": md,
                "source_url": f"https://mevzuat.adalet.gov.tr/ictihat/{doc_id}",
            })

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            f"Retrieval ({self.backend}): {len(decisions)} karar, {elapsed_ms:.0f}ms "
            f"(query_cache={cache_source})"
        )

        return RAGContext(
            question=question,
            decisions=decisions,
            total_found=len(decisions),
            search_time_ms=elapsed_ms,
            query_embedding=query_emb,  # v1.3.0+: ask() cache lookup için tekrar kullanır
            query_cache_hit=(cache_source != "miss"),  # v1.4.0+
            query_cache_source=cache_source,            # v1.4.0+
        )

    def _rerank_by_section(self, results: List, alpha: float = 0.15) -> List:
        """
        v1.5.0+ — Section-aware reranking.

        Türk hukuki kararlarda farklı bölümler farklı değer taşır:
          - GEREKÇE  → en yüksek (hukuki gerekçe, ilke)
          - HÜKÜM    → yüksek (somut karar)
          - KARAR    → orta (karar özeti)
          - ÖZET     → orta
          - DAVACI/DAVALI → düşük (taraflar, kimlik)
          - body     → değişmez (genel metin)

        Her sonuç için:
          new_score = original_score * (1 + alpha * section_weight)

        Bu, section boost'lu sonuçları üst sıraya çıkarır ama düşük
        cosine skorlu yüksek-section chunk'larını orijinal yüksek skorluların
        üstüne çıkarmaz (alpha=0.15 → max 15% boost).

        Args:
            results: List of (doc, score) tuples from ChromaDB search
            alpha: Boost factor (0.15 = max %15 boost)

        Returns:
            Reranked list of (doc, score) tuples
        """
        # Section weight tablosu
        SECTION_WEIGHTS = {
            "GEREKÇE": 1.0,    # en yüksek — hukuki ilke, gerekçe
            "HÜKÜM": 0.9,      # yüksek — somut karar
            "HUKUM": 0.9,      # ASCII fallback
            "KARAR": 0.7,      # orta — karar metni
            "ÖZET": 0.6,       # orta — özet
            "OZET": 0.6,       # ASCII fallback
            "TURKISH": 0.5,    # orta — Türkçe çeviri
            "BAŞLIK": 0.4,     # düşük — başlık
            "BASLIK": 0.4,     # ASCII fallback
            "DAVACI": 0.2,     # çok düşük — taraf kimliği
            "DAVALI": 0.2,     # çok düşük — taraf kimliği
            "İHBAR OLUNAN": 0.2,  # çok düşük
            "body": 0.0,       # değişmez — genel metin
            None: 0.0,         # section yok
        }

        def get_weight(section: Optional[str]) -> float:
            if not section:
                return 0.0
            return SECTION_WEIGHTS.get(section.upper(), 0.0)

        reranked = []
        boosts_applied = 0
        for doc, score in results:
            section = doc.metadata.get("section") if hasattr(doc, "metadata") else None
            weight = get_weight(section)
            if weight > 0:
                boosted = float(score) * (1.0 + alpha * weight)
                reranked.append((doc, boosted))
                boosts_applied += 1
            else:
                reranked.append((doc, float(score)))

        # Skora göre yeniden sırala (yüksek → düşük)
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Section-aware rerank: {boosts_applied}/{len(results)} chunk boost'landı "
            f"(alpha={alpha})"
        )
        return reranked

    async def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Tam RAG pipeline — retrieve + (cache lookup) + generate.

        v1.3.0+ flow:
            1. Retrieve top-K karar (NVIDIA query embed + ChromaDB search)
            2. Answer cache lookup (aynı query_emb ile ChromaDB qa_cache search)
               - Hit  → cache'den cevap döner, LLM çağrısı YAPILMAZ
               - Miss → LLM çağrısı yap, Q+A+citations cache'e yaz
            3. (Cache miss ise) LLM call
            4. Build RAGResponse (from_cache flag ile)

        Returns:
            RAGResponse — answer + citations + metadata (from_cache dahil)
        """
        from .prompts import SYSTEM_PROMPT_LEGAL, build_user_prompt, build_context_from_decisions

        t_total_start = time.time()

        # 1) Retrieve
        ctx = await self.retrieve(question, top_k=top_k)
        if not ctx.decisions:
            return RAGResponse(
                question=question,
                answer=(
                    "Bu soru için yüklü corpus'ta yeterince alakalı karar bulunamadı. "
                    "Lütfen daha geniş bir keyword ile `rag.load_corpora(...)` çağırın "
                    "veya farklı bir soru sorun."
                ),
                citations=[],
                context_decision_count=0,
                embedding_model=self._get_embedder().model if self._embedder else "n/a",
                llm_model="n/a",
                llm_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                total_time_ms=(time.time() - t_total_start) * 1000,
                retrieval_time_ms=ctx.search_time_ms,
                generation_time_ms=0,
            )

        # 2) v1.3.0+ — Answer cache lookup (query_emb'i reuse et)
        cache = self._get_answer_cache()
        if cache.enabled and ctx.query_embedding is not None:
            t_cache_start = time.time()
            hit = cache.lookup(ctx.query_embedding)
            cache_ms = (time.time() - t_cache_start) * 1000
            if hit is not None:
                logger.info(
                    f"Answer cache HIT — LLM çağrısı atlandı, cache lookup {cache_ms:.0f}ms"
                )
                response = RAGResponse(
                    question=question,
                    answer=hit.answer,
                    citations=hit.citations,
                    context_decision_count=len(hit.citations),
                    embedding_model=self._get_embedder().model,
                    llm_model=hit.llm_model,
                    llm_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    total_time_ms=(time.time() - t_total_start) * 1000,
                    retrieval_time_ms=ctx.search_time_ms,
                    generation_time_ms=cache_ms,
                    from_cache=True,
                    cache_score=hit.score,
                    llm_provider=hit.llm_provider,
                    query_cache_hit=ctx.query_cache_hit,
                    query_cache_source=ctx.query_cache_source,
                )
                logger.info(
                    f"RAG tamam (ANSWER CACHE) — toplam {response.total_time_ms:.0f}ms "
                    f"(retrieval {ctx.search_time_ms:.0f}ms [query_cache={ctx.query_cache_source}] "
                    f"+ answer_cache {cache_ms:.0f}ms, score={hit.score:.4f})"
                )
                return response

        # 3) Build context
        context_text = build_context_from_decisions(ctx.decisions)
        user_prompt = build_user_prompt(question, context_text)

        # 4) LLM call (cache miss)
        llm = self._get_llm_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(
            f"LLM çağrılıyor: provider={llm.provider}, model={llm.model}, "
            f"prompt={len(user_prompt)} chars"
        )
        t_llm_start = time.time()
        llm_resp = await llm.chat_async(
            messages=messages,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )
        gen_ms = (time.time() - t_llm_start) * 1000

        # 5) v1.3.0+ — Cache'e yaz (cache enabled ise)
        if cache.enabled and ctx.query_embedding is not None:
            cache.store(
                question=question,
                question_embedding=ctx.query_embedding,
                answer=llm_resp.text,
                citations=ctx.decisions,
                metadata={
                    "llm_model": llm.model,
                    "llm_provider": llm.provider,
                    "embedding_model": self._get_embedder().model,
                },
            )

        # 6) Build response
        response = RAGResponse(
            question=question,
            answer=llm_resp.text,
            citations=ctx.decisions,  # formatted_results formatı
            context_decision_count=len(ctx.decisions),
            embedding_model=self._get_embedder().model,
            llm_model=llm.model,
            llm_usage=llm_resp.usage,
            total_time_ms=(time.time() - t_total_start) * 1000,
            retrieval_time_ms=ctx.search_time_ms,
            generation_time_ms=gen_ms,
            from_cache=False,
            llm_provider=llm.provider,
            query_cache_hit=ctx.query_cache_hit,
            query_cache_source=ctx.query_cache_source,
        )
        logger.info(
            f"RAG tamam — toplam {response.total_time_ms:.0f}ms "
            f"(retrieval {ctx.search_time_ms:.0f}ms [query_cache={ctx.query_cache_source}] "
            f"+ gen {gen_ms:.0f}ms, provider={llm.provider}, "
            f"tokens: {llm_resp.usage.get('total_tokens', 0)})"
        )
        return response

    async def ask_stream(self, question: str, top_k: Optional[int] = None) -> AsyncIterator[str]:
        """
        Streaming RAG — LLM token'larını anında yield eder.

        Retrieval yapıldıktan sonra, cevap token token akıtılır.
        CLI/REPL ve FastAPI SSE için kullanılır.
        """
        from .prompts import SYSTEM_PROMPT_LEGAL, build_user_prompt, build_context_from_decisions

        # 1) Retrieve (non-stream)
        ctx = await self.retrieve(question, top_k=top_k)
        if not ctx.decisions:
            yield "Bu soru için yüklü corpus'ta yeterince alakalı karar bulunamadı."
            return

        # 2) Context + prompt
        context_text = build_context_from_decisions(ctx.decisions)
        user_prompt = build_user_prompt(question, context_text)

        # 3) Stream LLM
        llm = self._get_llm_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL},
            {"role": "user", "content": user_prompt},
        ]

        async for chunk in llm.chat_stream(
            messages=messages,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        ):
            yield chunk
