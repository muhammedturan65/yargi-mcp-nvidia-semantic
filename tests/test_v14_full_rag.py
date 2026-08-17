"""
v1.4.0 — Full RAG pipeline benchmark (NVIDIA LLM + query cache + answer cache).

Bu test 3 senaryoyu ölçer:
1. Tamamen cold (query cache MISS + answer cache MISS) — NVIDIA LLM çağrılır
2. Query cache HIT + answer cache MISS — retrieval hızlı, LLM çağrılır
3. Query cache HIT + answer cache HIT — hem retrieval hem LLM atlanır

Çalıştırma:
  python3 /home/z/my-project/scripts/test_v14_full_rag.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO = Path("/home/z/my-project/repos/yargi-mcp-nvidia-semantic")
sys.path.insert(0, str(REPO))

# NVIDIA config (embedding + LLM) — env var'dan oku (hardcoded değil)
_NVIDIA_KEY = os.environ.get("LOCAL_EMBEDDING_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("LOCAL_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
if _NVIDIA_KEY:
    os.environ.setdefault("LOCAL_EMBEDDING_API_KEY", _NVIDIA_KEY)
os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
os.environ.setdefault("LOCAL_EMBEDDING_DIMENSION", "4096")
os.environ.setdefault("LOCAL_EMBEDDING_INPUT_TYPE", "auto")
os.environ.setdefault("EMBEDDING_PROMPT_STYLE", "raw")
# NVIDIA LLM config
os.environ.setdefault("LLM_PROVIDER", "nvidia")
if _NVIDIA_KEY:
    os.environ.setdefault("LLM_API_KEY", _NVIDIA_KEY)
os.environ.setdefault("LLM_MODEL", "meta/llama-3.1-70b-instruct")
os.environ.setdefault("LLM_MAX_TOKENS", "1500")
# ChromaDB
os.environ["CHROMA_PERSIST_DIR"] = str(REPO / "chroma_db")
# RAG config
os.environ["RAG_BACKEND"] = "chroma"
os.environ["CHROMA_COLLECTION"] = "yargi_decisions"
os.environ["RAG_ANSWER_CACHE"] = "true"
os.environ["RAG_QUERY_CACHE"] = "true"


async def main():
    print("=" * 80)
    print("v1.4.0 FULL RAG PIPELINE BENCHMARK")
    print("  Query cache + Answer cache + NVIDIA LLM")
    print("=" * 80)

    from qa_rag import LegalQARAG

    rag = LegalQARAG(backend="chroma", llm_provider="nvidia")
    print(f"RAG init OK — backend={rag.backend}")

    # Cache'leri temizle ki gerçek cold-start ölçelim
    print("\nCache'ler temizleniyor (cold-start ölçümü için)...")
    qcache = rag._get_query_cache()
    acache = rag._get_answer_cache()
    qcache.clear()
    acache.clear()
    print(f"  Query cache: {qcache.size()} kayıt")
    print(f"  Answer cache: {acache.size()} kayıt")

    # Test sorusu — Türkçe hukuki
    question = "Muvazaalı tapu devrinde mirasçının açacağı dava nedir ve ispat yükü kimdedir?"

    # Senaryo 1: Cold start (her iki cache de MISS)
    print(f"\n{'='*80}")
    print(f"SENARYO 1: Cold start (query cache MISS + answer cache MISS)")
    print(f"{'='*80}")
    print(f"Soru: {question}")
    t0 = time.time()
    r1 = await rag.ask(question, top_k=5)
    t1 = time.time()
    print(f"\nCevap (ilk 300 char):")
    print(r1.answer[:300])
    print(f"\n---")
    print(f"Toplam süre: {r1.total_time_ms:.0f}ms ({(t1-t0):.1f}s)")
    print(f"  Retrieval: {r1.retrieval_time_ms:.0f}ms (query_cache={r1.query_cache_source})")
    print(f"  Generation: {r1.generation_time_ms:.0f}ms (LLM provider={r1.llm_provider})")
    print(f"  Tokens: {r1.llm_usage.get('total_tokens', 'N/A')}")
    print(f"  Answer cache: from_cache={r1.from_cache}")
    print(f"  Atıflar: {len(r1.citations)} adet")
    for c in r1.citations[:3]:
        print(f"    - {c}")

    # Senaryo 2: Query cache HIT + answer cache MISS (farklı soru)
    # Aynı soruyu sor ki answer cache de HIT olsun — senaryo 3 olur
    print(f"\n{'='*80}")
    print(f"SENARYO 2: Query cache HIT + Answer cache HIT (aynı soru tekrar)")
    print(f"{'='*80}")
    print(f"Soru: {question}")
    t0 = time.time()
    r2 = await rag.ask(question, top_k=5)
    t1 = time.time()
    print(f"\nCevap (ilk 300 char):")
    print(r2.answer[:300])
    print(f"\n---")
    print(f"Toplam süre: {r2.total_time_ms:.0f}ms ({(t1-t0):.1f}s)")
    print(f"  Retrieval: {r2.retrieval_time_ms:.0f}ms (query_cache={r2.query_cache_source})")
    print(f"  Generation: {r2.generation_time_ms:.0f}ms (LLM {'ATLANDI' if r2.from_cache else 'çağrıldı'})")
    print(f"  Answer cache: from_cache={r2.from_cache}, score={r2.cache_score:.4f}")
    print(f"  Atıflar: {len(r2.citations)} adet")

    # Senaryo 3: Yeni process simulation — persistent cache kullanımı
    print(f"\n{'='*80}")
    print(f"SENARYO 3: Yeni process (persistent cache HIT)")
    print(f"{'='*80}")
    print("Yeni LegalQARAG instance oluşturuluyor (LRU boş)...")
    rag2 = LegalQARAG(backend="chroma", llm_provider="nvidia")
    print(f"Soru: {question}")
    t0 = time.time()
    r3 = await rag2.ask(question, top_k=5)
    t1 = time.time()
    print(f"\nCevap (ilk 300 char):")
    print(r3.answer[:300])
    print(f"\n---")
    print(f"Toplam süre: {r3.total_time_ms:.0f}ms ({(t1-t0):.1f}s)")
    print(f"  Retrieval: {r3.retrieval_time_ms:.0f}ms (query_cache={r3.query_cache_source})")
    print(f"  Generation: {r3.generation_time_ms:.0f}ms (LLM {'ATLANDI' if r3.from_cache else 'çağrıldı'})")
    print(f"  Answer cache: from_cache={r3.from_cache}, score={r3.cache_score:.4f}")

    # Özet
    print(f"\n{'='*80}")
    print(f"ÖZET — v1.4.0 RAG Pipeline Benchmark")
    print(f"{'='*80}")
    print(f"{'Senaryo':<50} {'Süre':<12} {'Cache (Q/A)':<15}")
    print("-" * 80)
    print(f"{'1. Cold (NVIDIA query + NVIDIA LLM)':<50} {r1.total_time_ms/1000:<12.1f}s {'MISS/MISS':<15}")
    print(f"{'2. Warm (LRU query + answer cache HIT)':<50} {r2.total_time_ms/1000:<12.2f}s {f'{r2.query_cache_source.upper()}/HIT':<15}")
    print(f"{'3. Restart (persistent query + answer cache HIT)':<50} {r3.total_time_ms/1000:<12.2f}s {f'{r3.query_cache_source.upper()}/HIT':<15}")
    print(f"\nHızlanma:")
    print(f"  Warm vs Cold: {r1.total_time_ms / max(r2.total_time_ms, 1):.1f}x")
    print(f"  Restart vs Cold: {r1.total_time_ms / max(r3.total_time_ms, 1):.1f}x")

    # JSON sonuç
    results = {
        "test": "v1.4.0 full RAG pipeline benchmark",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "scenarios": {
            "cold_start": {
                "total_ms": round(r1.total_time_ms, 1),
                "retrieval_ms": round(r1.retrieval_time_ms, 1),
                "generation_ms": round(r1.generation_time_ms, 1),
                "query_cache_source": r1.query_cache_source,
                "answer_cache_hit": r1.from_cache,
                "tokens": r1.llm_usage.get("total_tokens"),
                "citations_count": len(r1.citations),
            },
            "warm_lru": {
                "total_ms": round(r2.total_time_ms, 1),
                "retrieval_ms": round(r2.retrieval_time_ms, 1),
                "generation_ms": round(r2.generation_time_ms, 1),
                "query_cache_source": r2.query_cache_source,
                "answer_cache_hit": r2.from_cache,
                "answer_cache_score": round(r2.cache_score, 4),
                "citations_count": len(r2.citations),
            },
            "restart_persistent": {
                "total_ms": round(r3.total_time_ms, 1),
                "retrieval_ms": round(r3.retrieval_time_ms, 1),
                "generation_ms": round(r3.generation_time_ms, 1),
                "query_cache_source": r3.query_cache_source,
                "answer_cache_hit": r3.from_cache,
                "answer_cache_score": round(r3.cache_score, 4),
            },
        },
        "speedup": {
            "warm_vs_cold": round(r1.total_time_ms / max(r2.total_time_ms, 1), 1),
            "restart_vs_cold": round(r1.total_time_ms / max(r3.total_time_ms, 1), 1),
        },
        "answer_preview": r1.answer[:500],
        "top_citations": [str(c) for c in r1.citations[:5]],
    }
    out_file = REPO / "tests" / "v14_full_rag_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSonuçlar: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
