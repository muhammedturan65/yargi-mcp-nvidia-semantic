"""
v1.4.0 — Query embedding cache smoke test.

Smoke testleri:
1. Import testleri (yeni exportlar)
2. normalize_query doğru çalışıyor mu
3. QueryEmbeddingCache init (ChromaDB collection oluşturma/getirme)
4. Store + lookup (LRU + persistent)
5. Cache hit/miss istatistikleri
6. LegalQARAG retrieve() — ilk sorgu MISS, ikinci sorgu LRU HIT

Çalıştırma:
  python3 /home/z/my-project/scripts/test_v14_query_cache.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project/repos/yargi-mcp-nvidia-semantic")
sys.path.insert(0, str(REPO))

# NVIDIA API key — env var'dan oku (hardcoded değil)
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LOCAL_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
# LOCAL_EMBEDDING_API_KEY env'ten beklenir
os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
os.environ.setdefault("LOCAL_EMBEDDING_DIMENSION", "4096")
os.environ.setdefault("LOCAL_EMBEDDING_INPUT_TYPE", "auto")
os.environ.setdefault("EMBEDDING_PROMPT_STYLE", "raw")
# ChromaDB
os.environ["CHROMA_PERSIST_DIR"] = str(REPO / "chroma_db")
# RAG config — Chroma backend + answer cache + query cache
os.environ["RAG_BACKEND"] = "chroma"
os.environ["CHROMA_COLLECTION"] = "yargi_decisions"
os.environ["RAG_ANSWER_CACHE"] = "true"
os.environ["RAG_QUERY_CACHE"] = "true"


def test_imports():
    """1. Import testleri."""
    print("\n[1/6] Import testleri...")
    from qa_rag import (
        LegalQARAG, RAGResponse, RAGContext,
        LLMClient, NvidiaLLMClient, get_llm_client, LLMResponse,
        AnswerCache, CacheHit,
        QueryEmbeddingCache, QueryCacheHit,
        normalize_query, cache_key,
        SYSTEM_PROMPT_LEGAL, build_user_prompt,
        format_citations, Citation,
        LegalChunker, Chunk, chunk_text,
        BedestenIndexer, IndexResult, IndexProgress,
    )
    print("  ✓ Tüm importlar başarılı (16 symbol)")

    import qa_rag
    assert qa_rag.__version__ == "1.4.0", f"Beklenen 1.4.0, got {qa_rag.__version__}"
    print(f"  ✓ __version__ = {qa_rag.__version__}")
    return True


def test_normalize_query():
    """2. normalize_query doğru çalışıyor mu."""
    print("\n[2/6] normalize_query testi...")
    from qa_rag import normalize_query, cache_key

    # Türkçe karakterler
    assert normalize_query("Mirasçı hangi davayı açar?") == "mirasci hangi davayi acar"
    assert normalize_query("  mirasçı HANGİ davaYI açar ") == "mirasci hangi davayi acar"
    assert normalize_query("Muvazaa ispat yükü kimde?") == "muvazaa ispat yuku kimde"
    assert normalize_query("") == ""
    assert normalize_query("   ") == ""

    # Cache key deterministic
    k1 = cache_key("Mirasçı hangi davayı açar?")
    k2 = cache_key("mirasçi hangi davayi açar?")
    assert k1 == k2, f"Aynı normalize sonuc için farklı key: {k1} vs {k2}"
    assert len(k1) == 16

    print(f"  ✓ normalize_query: Türkçe→ASCII, lowercase, punct strip OK")
    print(f"  ✓ cache_key: deterministic 16-char hex (örnek: {k1})")
    return True


def test_cache_init():
    """3. QueryEmbeddingCache init."""
    print("\n[3/6] QueryEmbeddingCache init...")
    from qa_rag import QueryEmbeddingCache

    cache = QueryEmbeddingCache(dimension=4096)
    assert cache.enabled, "Cache enable=false ama default true olmalı"
    assert cache.dimension == 4096
    print(f"  ✓ Cache aktif — collection='{cache.collection_name}', lru_size={cache.lru_size}")
    print(f"  ✓ Persistent kayıt sayısı: {cache.size()}")
    return cache


def test_store_lookup(cache):
    """4. Store + lookup (LRU + persistent)."""
    print("\n[4/6] Store + lookup testi...")
    import numpy as np

    # Önce cache'i temizle (temiz state ile başla)
    print("  Cache temizleniyor (test için)...")
    cache.clear()
    assert cache.size() == 0
    assert cache.lru_size_current() == 0

    # Test verisi
    q1 = "Mirasçı hangi davayı açar?"
    q1_norm = "mirasci hangi davayi acar"
    emb1 = np.random.randn(4096).astype(np.float32)
    emb1 = emb1 / np.linalg.norm(emb1)  # L2 normalize

    q2 = "Muvazaa ispat yükü kimde?"
    emb2 = np.random.randn(4096).astype(np.float32)
    emb2 = emb2 / np.linalg.norm(emb2)

    # Store q1
    key1 = cache.store(q1, emb1)
    assert len(key1) == 16
    assert cache.lru_size_current() == 1
    assert cache.size() == 1
    print(f"  ✓ Store q1: key={key1}, lru={cache.lru_size_current()}, persistent={cache.size()}")

    # Lookup q1 — LRU HIT olmalı
    hit1 = cache.lookup(q1)
    assert hit1 is not None
    assert hit1.source == "lru"
    assert np.allclose(hit1.embedding, emb1, atol=1e-5)
    print(f"  ✓ Lookup q1: LRU HIT, source={hit1.source}")

    # Store q2
    key2 = cache.store(q2, emb2)
    assert cache.lru_size_current() == 2
    assert cache.size() == 2
    print(f"  ✓ Store q2: key={key2}")

    # Lookup q2 — LRU HIT
    hit2 = cache.lookup(q2)
    assert hit2 is not None
    assert hit2.source == "lru"
    print(f"  ✓ Lookup q2: LRU HIT")

    # Farklı sorgu — MISS
    hit_miss = cache.lookup("Tapu iptal süresi nedir?")
    assert hit_miss is None
    print(f"  ✓ Lookup yeni soru: MISS (expected)")

    # Stats
    stats = cache.get_stats()
    assert stats["hits_lru"] == 2
    assert stats["misses"] == 1
    assert stats["stores"] == 2
    print(f"  ✓ Stats: {stats['hits_lru']} LRU hits, {stats['misses']} misses, {stats['stores']} stores")

    return True


def test_persistence(cache):
    """5. Persistent cache: yeni instance, eski kayıtlar."""
    print("\n[5/6] Persistence testi (yeni instance)...")
    import numpy as np
    from qa_rag import QueryEmbeddingCache

    # Yeni instance — persistent collection'dan yüklenmeli
    cache2 = QueryEmbeddingCache(dimension=4096)
    print(f"  Yeni instance persistent kayıt: {cache2.size()}")
    assert cache2.size() >= 2, "Yeni instance persistent kayıtları görmeli"

    # Eski sorgu — LRU boş, persistent HIT olmalı
    hit = cache2.lookup("Mirasçı hangi davayı açar?")
    assert hit is not None
    assert hit.source == "persistent"
    print(f"  ✓ Persistent HIT (yeni instance): source={hit.source}")

    # Şimdi LRU'da da olmalı
    hit2 = cache2.lookup("Mirasçı hangi davayı açar?")
    assert hit2 is not None
    assert hit2.source == "lru"
    print(f"  ✓ İkinci lookup: LRU HIT (persistent'tan taşındı)")

    return True


def test_rag_retrieve_with_cache():
    """6. LegalQARAG retrieve() — gerçek NVIDIA çağrısı + cache."""
    print("\n[6/6] LegalQARAG retrieve() ile query cache testi...")
    import asyncio
    from qa_rag import LegalQARAG, QueryEmbeddingCache

    rag = LegalQARAG(backend="chroma", llm_provider="nvidia")
    print(f"  RAG init: backend={rag.backend}")

    # Test öncesi cache'i temizle ki gerçek MISS görelim
    print("  Query cache temizleniyor (test için)...")
    cache = rag._get_query_cache()
    cache.clear()

    async def run_test():
        # Benzersiz bir sorgu kullan — cache'de kesinlikle yok
        q1 = "Muvazaalı tapu devrinde mirasçının hak kaybına uğraması halinde açacağı dava nedir?"
        print(f"\n  Q1 (ilk sorgu, expected MISS): {q1[:70]}...")
        ctx1 = await rag.retrieve(q1, top_k=5)
        print(f"  → {len(ctx1.decisions)} karar, {ctx1.search_time_ms:.0f}ms")
        print(f"  → query_cache_hit={ctx1.query_cache_hit}, source={ctx1.query_cache_source}")
        assert not ctx1.query_cache_hit, "İlk sorgu MISS olmalı"
        assert ctx1.query_cache_source == "miss"

        # İkinci sorgu (aynı) — LRU HIT olacak
        print(f"\n  Q2 (aynı sorgu, expected LRU HIT): {q1[:70]}...")
        ctx2 = await rag.retrieve(q1, top_k=5)
        print(f"  → {len(ctx2.decisions)} karar, {ctx2.search_time_ms:.0f}ms")
        print(f"  → query_cache_hit={ctx2.query_cache_hit}, source={ctx2.query_cache_source}")
        assert ctx2.query_cache_hit, "İkinci sorgu LRU HIT olmalı"
        assert ctx2.query_cache_source == "lru"

        # Hız karşılaştırması
        speedup = ctx1.search_time_ms / max(ctx2.search_time_ms, 0.01)
        print(f"\n  ⚡ Hızlanma: {speedup:.1f}x")
        print(f"     İlk sorgu (NVIDIA):  {ctx1.search_time_ms:.0f}ms")
        print(f"     Tekrar (LRU HIT):    {ctx2.search_time_ms:.0f}ms")

        # Yeni instance — persistent HIT testi
        print(f"\n  Q3 (yeni instance, expected persistent HIT): {q1[:70]}...")
        rag2 = LegalQARAG(backend="chroma", llm_provider="nvidia")
        ctx3 = await rag2.retrieve(q1, top_k=5)
        print(f"  → {len(ctx3.decisions)} karar, {ctx3.search_time_ms:.0f}ms")
        print(f"  → query_cache_hit={ctx3.query_cache_hit}, source={ctx3.query_cache_source}")
        # Yeni instance'ın LRU'su boş, persistent'ten gelmeli
        assert ctx3.query_cache_hit, "Yeni instance persistent HIT olmalı"
        assert ctx3.query_cache_source == "persistent"

        return {
            "first_query_ms": round(ctx1.search_time_ms, 1),
            "second_query_ms_lru": round(ctx2.search_time_ms, 1),
            "third_query_ms_persistent": round(ctx3.search_time_ms, 1),
            "speedup_lru": round(speedup, 1),
            "speedup_persistent": round(ctx1.search_time_ms / max(ctx3.search_time_ms, 0.01), 1),
            "first_query_results": len(ctx1.decisions),
            "second_query_results": len(ctx2.decisions),
            "third_query_results": len(ctx3.decisions),
            "top_1_first": ctx1.decisions[0].get("title", "")[:80] if ctx1.decisions else "",
            "top_1_score_first": round(ctx1.decisions[0].get("score", 0), 4) if ctx1.decisions else 0,
        }

    return asyncio.run(run_test())


def main():
    print("=" * 80)
    print("v1.4.0 QUERY EMBEDDING CACHE SMOKE TEST")
    print("=" * 80)
    print(f"Repo: {REPO}")
    print(f"Python: {sys.executable}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    try:
        test_imports()
        results["imports"] = "PASS"
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        results["imports"] = f"FAIL: {e}"
        return results

    try:
        test_normalize_query()
        results["normalize_query"] = "PASS"
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        results["normalize_query"] = f"FAIL: {e}"
        return results

    try:
        cache = test_cache_init()
        results["cache_init"] = "PASS"
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        results["cache_init"] = f"FAIL: {e}"
        return results

    try:
        test_store_lookup(cache)
        results["store_lookup"] = "PASS"
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        results["store_lookup"] = f"FAIL: {e}"
        import traceback
        traceback.print_exc()
        return results

    try:
        test_persistence(cache)
        results["persistence"] = "PASS"
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        results["persistence"] = f"FAIL: {e}"
        import traceback
        traceback.print_exc()

    try:
        rag_results = test_rag_retrieve_with_cache()
        results["rag_retrieve"] = "PASS"
        results["rag_retrieve_metrics"] = rag_results
    except Exception as e:
        print(f"  ✗ HATA: {e}")
        results["rag_retrieve"] = f"FAIL: {e}"
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("ÖZET")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k}: {v}")

    # Sonuçları JSON'a kaydet
    import json
    out_file = REPO / "tests" / "v14_query_cache_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test": "v1.4.0 query embedding cache smoke test",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSonuçlar: {out_file}")

    return results


if __name__ == "__main__":
    main()
