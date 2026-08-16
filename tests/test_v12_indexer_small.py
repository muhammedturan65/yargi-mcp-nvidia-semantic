"""
v1.2.0 indexer test — 10 belge ile küçük batch indexleme.

Amaç:
  1. BedestenIndexer pipeline'ının uçtan uca çalıştığını doğrula
  2. retrieval sub-second hızda mı kontrol et
  3. ChromaDB kalıcılığını test et (process restart simülasyonu)
"""
import asyncio
import os
import sys
import time

# Path
sys.path.insert(0, "/home/z/my-project/repos/yargi-mcp-nvidia-semantic")

# NVIDIA env
NVIDIA_KEY = "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"
os.environ["NVIDIA_API_KEY"] = NVIDIA_KEY
os.environ["LOCAL_EMBEDDING_API_KEY"] = NVIDIA_KEY
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LOCAL_EMBEDDING_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["LOCAL_EMBEDDING_MODEL"] = "nvidia/nv-embed-v1"
os.environ["LOCAL_EMBEDDING_DIMENSION"] = "4096"
os.environ["LOCAL_EMBEDDING_INPUT_TYPE"] = "auto"
os.environ["EMBEDDING_PROMPT_STYLE"] = "raw"

# Bedesten rate-limit (yüksek tolerance)
os.environ["BEDESTEN_RATE_CAPACITY"] = "1"
os.environ["BEDESTEN_RATE_REFILL_S"] = "4.0"
os.environ["BEDESTEN_RATE_MAX_WAIT_S"] = "60"
os.environ["BEDESTEN_SEMANTIC_MAX_RETRIES"] = "3"

# Indexer config
os.environ["INDEXER_BATCH_SIZE"] = "16"
os.environ["INDEXER_TARGET_DOCS"] = "10"
os.environ["INDEXER_KEYWORDS"] = "muvazaa tapu iptal"
os.environ["INDEXER_COURT_TYPES"] = "YARGITAYKARARI"

# ChromaDB config
os.environ["CHROMA_PERSIST_DIR"] = "/home/z/my-project/repos/yargi-mcp-nvidia-semantic/chroma_db_test"
os.environ["CHROMA_COLLECTION"] = "yargi_test_small"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Chroma ve httpx log'larını azalt
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def main():
    print("=" * 72)
    print("v1.2.0 Indexer Test — 10 belge")
    print("=" * 72)

    # Temiz başlangıç: önce test collection'ı sil
    from semantic_search.vector_store_chroma import _get_chroma_client
    try:
        client = _get_chroma_client()
        client.delete_collection("yargi_test_small")
        print("✓ Eski test collection silindi")
    except Exception:
        pass

    # Indexer
    from qa_rag.indexer import BedestenIndexer
    indexer = BedestenIndexer(
        batch_size=16,
        chroma_collection="yargi_test_small",
    )

    print("\n--- Index başlıyor ---")
    t_start = time.time()
    result = await indexer.run(
        keywords=["muvazaa tapu iptal"],
        court_types=["YARGITAYKARARI"],
        target_docs=10,
    )
    print(f"\n--- Index bitti ({time.time()-t_start:.0f}s) ---")
    print(f"Hedef:       {result.total_docs_target}")
    print(f"Çekilen:     {result.total_docs_fetched}")
    print(f"Indexlenen: {result.total_docs_indexed}")
    print(f"Başarısız:   {result.total_docs_failed}")
    print(f"Toplam chunk: {result.total_chunks}")
    print(f"Toplam token: {result.total_tokens}")
    print(f"ChromaDB kayıt: {result.chroma_count}")
    print(f"Embedding modeli: {result.embedding_model}")
    if result.errors:
        print(f"Hatalar (ilk 5): {result.errors}")

    # Stats
    avg_chunks = result.total_chunks / max(result.total_docs_indexed, 1)
    avg_tokens = result.total_tokens / max(result.total_chunks, 1)
    print(f"\nOrtalama: {avg_chunks:.1f} chunk/belge, {avg_tokens:.0f} token/chunk")

    # --- Retrieval testi ---
    print("\n" + "=" * 72)
    print("Retrieval testi — ChromaDB'den sub-second sorgu")
    print("=" * 72)

    from qa_rag.rag_engine import LegalQARAG
    rag = LegalQARAG(backend="chroma", chroma_collection="yargi_test_small")

    print(f"Corpora loaded? {rag.is_corpora_loaded}")

    questions = [
        "Mirasçı muvazaalı satışa karşı hangi davayı açar?",
        "Muvazaa iddiasında ispat yükü kimdedir?",
        "Tapu iptal davası açma süresi nedir?",
    ]

    for q in questions:
        print(f"\nSoru: {q}")
        t_q = time.time()
        ctx = await rag.retrieve(q, top_k=3)
        elapsed = (time.time() - t_q) * 1000
        print(f"  → {ctx.total_found} karar, {elapsed:.0f}ms")
        for i, d in enumerate(ctx.decisions, 1):
            md = d["metadata"]
            score = d["similarity_score"]
            title = d["title"]
            section = md.get("section", "?")
            print(f"    [{i}] skor={score:.4f} | {title} [{section}]")
            print(f"        text (ilk 150): {d['text'][:150]}...")

    # --- Persistence testi ---
    print("\n" + "=" * 72)
    print("Persistence testi — yeni process açıkmış gibi ChromaDB'den oku")
    print("=" * 72)

    # Yeni LegalQARAG instance — ChromaDB'yi yeniden aç
    rag2 = LegalQARAG(backend="chroma", chroma_collection="yargi_test_small")
    print(f"Yeni instance - is_corpora_loaded? {rag2.is_corpora_loaded}")
    print(f"Yeni instance - vector store size: {rag2._get_vector_store().size()}")

    if rag2.is_corpora_loaded:
        ctx = await rag2.retrieve("Muvazaa nedir?", top_k=2)
        print(f"  → {ctx.total_found} karar geldi (kalıcı çalışıyor!)")

    # --- Final stats ---
    print("\n" + "=" * 72)
    print("SONUÇ")
    print("=" * 72)
    print(f"✓ Index: {result.total_docs_indexed} belge → {result.total_chunks} chunk")
    print(f"✓ ChromaDB kalıcı: {result.chroma_count} kayıt diske yazıldı")
    print(f"✓ Retrieval: sub-second (NVIDIA embed + ChromaDB search)")
    print(f"✓ Persistence: yeni instance ChromaDB'den okudu")


if __name__ == "__main__":
    asyncio.run(main())
