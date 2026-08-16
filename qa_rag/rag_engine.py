"""
RAG engine — yargi-mcp semantik arama + NVIDIA LLM entegrasyonu.

Kullanıcı sorusu → embedding → vector store'da top-K arama →
context construction → NVIDIA LLM ile cevap üretimi →
atıflı yanıt.

İki mod var:
    - "loaded":    Önceden yüklenmiş vector store'u kullanır (hızlı, demo modu)
    - "fresh":     Her soruda search_bedesten_semantic çağırır (yavaş, taze veri)

Demo için "loaded" modu önerilir. Üretim için ChromaDB kalıcı store önerilir
(sonraki milestone).
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


class LegalQARAG:
    """
    Hukuki QA RAG pipeline.

    Usage:
        rag = LegalQARAG()
        await rag.load_corpora("muvazaa tapu iptal")  # 30 karar yükle (~2 dk)
        response = await rag.ask("Mirasçı hangi davayı açar?")
        print(response.answer)
    """

    def __init__(
        self,
        n_decisions_per_query: int = 30,  # search_bedesten_semantic batch size
        top_k_retrieval: int = 5,         # LLM'e kaç karar feed'lenecek
        llm_temperature: float = 0.2,     # hukuki: düşük yaratıcılık
        llm_max_tokens: int = 1500,
    ):
        self.n_decisions_per_query = n_decisions_per_query
        self.top_k_retrieval = top_k_retrieval
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens

        # Lazy-loaded components
        self._embedder = None
        self._vector_store = None
        self._llm_client = None
        self._mcp_module = None
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
            from semantic_search.vector_store import VectorStore
            embedder = self._get_embedder()
            self._vector_store = VectorStore(dimension=embedder.dimension)
            logger.info(f"VectorStore oluşturuldu (dimension={embedder.dimension})")
        return self._vector_store

    def _get_llm_client(self):
        if self._llm_client is None:
            from .llm_client import NvidiaLLMClient
            self._llm_client = NvidiaLLMClient(
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
            logger.info(f"LLM client yüklendi: {self._llm_client.model}")
        return self._llm_client

    def _get_mcp_module(self):
        """mcp_server_main modülünü lazy yükle (search_bedesten_semantic için)."""
        if self._mcp_module is None:
            import mcp_server_main
            self._mcp_module = mcp_server_main
            logger.info("mcp_server_main yüklendi (Bedesten semantic search için)")
        return self._mcp_module

    # ---------- Public API ----------

    @property
    def is_corpora_loaded(self) -> bool:
        """Vector store'da en az 1 karar var mı?"""
        if self._vector_store is None:
            return False
        return self._vector_store.size() > 0

    async def load_corpora(
        self,
        initial_keyword: str = "muvazaa tapu iptal",
        semantic_query: str = "Mirasçının muvazaalı satış işlemine karşı tapu iptali ve tescil davası açması",
        court_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Bedesten API'den karar çekip vector store'a yükle.

        Bu, search_bedesten_semantic tool'unu çağırır. ~2 dk sürer (rate-limit).
        Tekrar çağrılırsa mevcut kararları temizleyip yeniden yükler.

        Returns:
            search_bedesten_semantic'in döndüğü dict (status, results, stats, ...)
        """
        if court_types is None:
            court_types = ["YARGITAYKARARI"]

        mcp = self._get_mcp_module()
        if not getattr(mcp, "SEMANTIC_SEARCH_AVAILABLE", False):
            raise RuntimeError(
                "yargi-mcp semantik arama aktif değil. NVIDIA env vars'ları kontrol edin."
            )

        logger.info(f"Corpora yükleniyor: keyword='{initial_keyword}'")
        result = await mcp.search_bedesten_semantic(
            initial_keyword=initial_keyword,
            query=semantic_query,
            court_types=court_types,
            top_k=self.n_decisions_per_query,
        )

        if result.get("status") != "success":
            raise RuntimeError(f"Bedesten search hatası: {result.get('message', result)}")

        # search_bedesten_semantic kendi vector_store'una ekledi.
        # Bizim LegalQARAG vector store'umuz ayrı olduğu için, mcp_server_main'in
        # vector_store instance'ını referans al.
        # (İleride refactor: shared singleton vector store)
        self._vector_store = getattr(mcp, "_qa_rag_vector_store", None) or self._vector_store

        # mcp_server_main vector_store'unu al (search_bedesten_semantic oraya ekledi)
        # mcp_server_main içinde local bir vector_store değişkeni var, function-scope.
        # Bu durumda en kolay yol: sonuçtaki decision listesini kendi store'umuza eklemek.
        if self._vector_store.size() == 0:
            # mcp_server_main'in local store'u erişilebilir değil.
            # Çözüm: search_bedesten_semantic'in döndürdüğü results listesinden
            #        tekrar embed edip kendi store'umuza ekleyelim.
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

        Returns:
            RAGContext — decisions listesi formatted_results formatında
        """
        from .prompts import build_context_from_decisions  # forward decl

        t0 = time.time()

        if not self.is_corpora_loaded:
            raise RuntimeError(
                "Corpora yüklenmemiş. Önce rag.load_corpora() çağırın."
            )

        embedder = self._get_embedder()
        vs = self._vector_store

        # Query embedding
        query_emb = embedder.encode_query(question, task="legal question answering")
        if isinstance(query_emb, np.ndarray) and query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Search
        k = top_k or self.top_k_retrieval
        results = vs.search(query_emb[0] if hasattr(query_emb, '__getitem__') else query_emb,
                            top_k=k)

        decisions = []
        for doc, score in results:
            title_parts = []
            md = doc.metadata
            if md.get("birim_adi"):
                title_parts.append(md["birim_adi"])
            if md.get("esas_no"):
                title_parts.append(f"Esas: {md['esas_no']}")
            if md.get("karar_no"):
                title_parts.append(f"Karar: {md['karar_no']}")
            if md.get("karar_tarihi"):
                title_parts.append(f"Tarih: {md['karar_tarihi']}")

            decisions.append({
                "document_id": doc.id,
                "title": " - ".join(title_parts) if title_parts else f"Document {doc.id}",
                "similarity_score": float(score),
                "preview": doc.text[:500] + "..." if len(doc.text) > 500 else doc.text,
                "text": doc.text,
                "metadata": md,
                "source_url": f"https://mevzuat.adalet.gov.tr/ictihat/{doc.id}",
            })

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(f"Retrieval: {len(decisions)} karar, {elapsed_ms:.0f}ms")

        return RAGContext(
            question=question,
            decisions=decisions,
            total_found=len(decisions),
            search_time_ms=elapsed_ms,
        )

    async def ask(self, question: str, top_k: Optional[int] = None) -> RAGResponse:
        """
        Tam RAG pipeline — retrieve + generate.

        Returns:
            RAGResponse — answer + citations + metadata
        """
        from .prompts import SYSTEM_PROMPT_LEGAL, build_user_prompt, build_context_from_decisions
        from .citations import build_citations_from_decisions

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

        # 2) Build context
        context_text = build_context_from_decisions(ctx.decisions)
        user_prompt = build_user_prompt(question, context_text)

        # 3) LLM call
        llm = self._get_llm_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_LEGAL},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"LLM çağrılıyor: {llm.model}, prompt={len(user_prompt)} chars")
        t_llm_start = time.time()
        llm_resp = await llm.chat_async(
            messages=messages,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
        )
        gen_ms = (time.time() - t_llm_start) * 1000

        # 4) Build response
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
        )
        logger.info(
            f"RAG tamam — toplam {response.total_time_ms:.0f}ms "
            f"(retrieval {ctx.search_time_ms:.0f}ms + gen {gen_ms:.0f}ms, "
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
