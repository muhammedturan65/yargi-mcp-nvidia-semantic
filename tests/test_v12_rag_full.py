"""
v1.2.0 RAG pipeline test — mevcut ChromaDB ile tam RAG çağrısı.

Index arka planda devam ederken, şu anki ~70 chunk (~17 belge) ile:
  1. Retrieval benchmark
  2. Tam RAG (LLM çağrısı dahil) ile 2 hukuki soru
  3. v1.1.0 ile hız/kalite karşılaştırması
"""
import asyncio
import os
import sys
import time
import json

sys.path.insert(0, "/home/z/my-project/repos/yargi-mcp-nvidia-semantic")

NVIDIA_KEY = "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"
os.environ["NVIDIA_API_KEY"] = NVIDIA_KEY
os.environ["LOCAL_EMBEDDING_API_KEY"] = NVIDIA_KEY
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LOCAL_EMBEDDING_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["LOCAL_EMBEDDING_MODEL"] = "nvidia/nv-embed-v1"
os.environ["LOCAL_EMBEDDING_DIMENSION"] = "4096"
os.environ["LOCAL_EMBEDDING_INPUT_TYPE"] = "auto"
os.environ["EMBEDDING_PROMPT_STYLE"] = "raw"

os.environ["CHROMA_PERSIST_DIR"] = "/home/z/my-project/repos/yargi-mcp-nvidia-semantic/chroma_db_v12"
os.environ["CHROMA_COLLECTION"] = "yargi_v12_medium"

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


async def main():
    print("=" * 72)
    print("v1.2.0 RAG Pipeline Test — Tam LLM çağrısı dahil")
    print("=" * 72)

    from qa_rag.rag_engine import LegalQARAG
    rag = LegalQARAG(
        backend="chroma",
        chroma_collection="yargi_v12_medium",
        top_k_retrieval=5,
        llm_temperature=0.2,
        llm_max_tokens=1500,
    )

    vs = rag._get_vector_store()
    print(f"ChromaDB kayıt: {vs.size()} chunk")
    print(f"Embedding modeli: {rag._get_embedder().model}")
    print(f"LLM modeli: {rag._get_llm_client().model}")
    print(f"Backend: {rag.backend}")

    # --- Retrieval benchmark ---
    print(f"\n{'='*72}")
    print("1) RETRIEVAL BENCHMARK")
    print(f"{'='*72}")

    questions = [
        "Mirasçı muvazaalı satışa karşı hangi davayı açar?",
        "Muvazaa iddiasında ispat yükü kimdedir?",
        "Tapu iptal davası açma süresi nedir?",
        "Muris muvazaası nedir ve nasıl ispatlanır?",
        "Tapu iptal ve tescil davasında görevli mahkeme",
    ]

    retrieval_results = []
    total_ms = 0
    for q in questions:
        t = time.time()
        ctx = await rag.retrieve(q, top_k=5)
        elapsed_ms = (time.time() - t) * 1000
        total_ms += elapsed_ms
        top_score = ctx.decisions[0]["similarity_score"] if ctx.decisions else 0
        print(f"\nQ: {q}")
        print(f"  → {ctx.total_found} karar, {elapsed_ms:.0f}ms, top skor={top_score:.4f}")
        for i, d in enumerate(ctx.decisions[:3], 1):
            md = d["metadata"]
            title_short = f"{md.get('birim_adi', '?')} E.{md.get('esas_no', '?')}"
            section = md.get("section", "")
            sec_str = f" [{section}]" if section and section != "body" else ""
            print(f"    [{i}] {d['similarity_score']:.4f} | {title_short}{sec_str}")

        retrieval_results.append({
            "question": q,
            "elapsed_ms": elapsed_ms,
            "num_results": ctx.total_found,
            "top_score": top_score,
        })

    avg_ms = total_ms / len(questions)
    print(f"\nRetrieval Ortalama: {avg_ms:.0f}ms/sorgu (NVIDIA query embed + ChromaDB search)")

    # --- Tam RAG pipeline ---
    print(f"\n{'='*72}")
    print("2) TAM RAG PIPELINE (LLM çağrısı dahil)")
    print(f"{'='*72}")

    qa_pairs = [
        ("Mirasçı muvazaalı satışa karşı hangi davayı açar?", "tapu iptal ve tescil davası"),
        ("Muvazaa iddiasında ispat yükü kimdedir?", "iddia sahibi taraf"),
    ]

    rag_results = []
    for q, expected_keyword in qa_pairs:
        print(f"\n--- Soru: {q}")
        print(f"    (Beklenen anahtar kelime: '{expected_keyword}')")

        t = time.time()
        response = await rag.ask(q, top_k=5)
        elapsed = time.time() - t

        print(f"\nRetrieval: {response.retrieval_time_ms:.0f}ms")
        print(f"LLM:       {response.generation_time_ms:.0f}ms")
        print(f"Tokens:    {response.llm_usage}")
        print(f"Toplam:    {elapsed:.1f}s")
        print(f"\nCEVAP:")
        print(response.answer)

        print(f"\nAtıflar ({len(response.citations)}):")
        for i, c in enumerate(response.citations, 1):
            md = c["metadata"]
            print(f"  [{i}] {md.get('birim_adi', '?')} "
                  f"E.{md.get('esas_no', '?')} "
                  f"K.{md.get('karar_no', '?')} "
                  f"skor={c['similarity_score']:.4f}")

        # Basit semantic check: beklenen keyword cevapta var mı?
        answer_lower = response.answer.lower()
        expected_lower = expected_keyword.lower()
        has_expected = expected_lower in answer_lower or expected_keyword in response.answer

        print(f"\n✓ Beklenen keyword cevapta {'var' if has_expected else 'YOK'}: '{expected_keyword}'")

        rag_results.append({
            "question": q,
            "expected_keyword": expected_keyword,
            "has_expected_keyword": has_expected,
            "total_time_s": elapsed,
            "retrieval_ms": response.retrieval_time_ms,
            "generation_ms": response.generation_time_ms,
            "llm_usage": response.llm_usage,
            "llm_model": response.llm_model,
            "embedding_model": response.embedding_model,
            "citations_count": len(response.citations),
            "answer_length": len(response.answer),
            "answer_preview": response.answer[:300],
            "citations": [
                {
                    "birim": c["metadata"].get("birim_adi"),
                    "esas": c["metadata"].get("esas_no"),
                    "karar": c["metadata"].get("karar_no"),
                    "score": c["similarity_score"],
                }
                for c in response.citations
            ],
        })

    # --- Final summary ---
    print(f"\n{'='*72}")
    print("v1.2.0 RAG TEST SONUCU")
    print(f"{'='*72}")

    chroma_count = vs.size()
    print(f"ChromaDB:    {chroma_count} chunk indexli")
    print(f"Retrieval:   {avg_ms:.0f}ms ort (NVIDIA query embed dahil)")
    print(f"             v1.1.0'da retrieval ~2 dk (Bedesten fetch + embed + search)")
    print(f"             v1.2.0'da ChromaDB search ~5-20ms, NVIDIA embed ~{avg_ms-15:.0f}ms")
    print(f"             Hız artışı: ~{(120000/avg_ms):.0f}x")
    print()
    print(f"RAG pipeline ({len(qa_pairs)} soru):")
    for r in rag_results:
        status = "✓ PASS" if r["has_expected_keyword"] else "✗ FAIL"
        print(f"  {status} | {r['total_time_s']:.1f}s | {r['question'][:50]}")
        print(f"         retrieval={r['retrieval_ms']:.0f}ms, llm={r['generation_ms']:.0f}ms, tokens={r['llm_usage'].get('total_tokens', 0)}")

    # JSON output
    summary = {
        "version": "1.2.0",
        "test_type": "rag_pipeline",
        "chroma_count": chroma_count,
        "retrieval_benchmark": {
            "avg_ms": avg_ms,
            "results": retrieval_results,
        },
        "rag_results": rag_results,
        "speedup_vs_v1_1_0": 120000 / avg_ms,  # v1.1.0 ~120s vs v1.2.0 avg_ms
    }

    out_path = "/home/z/my-project/scripts/v12_rag_test_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
