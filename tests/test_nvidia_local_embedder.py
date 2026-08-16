#!/usr/bin/env python3
"""
NVIDIA API'yi yargi-mcp'nin LocalEmbedder'ı üzerinden test eder.
Patched LocalEmbedder artık NVIDIA'nın asimetrik modellerini (input_type) destekliyor.

Çalıştırma:
    python /home/z/my-project/scripts/test_nvidia_local_embedder.py
"""
import os
import sys
import json
import time
import numpy as np

# NVIDIA env vars — API key'i env var'dan al (hardcoded değil)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LOCAL_EMBEDDING_API_KEY")
if not NVIDIA_API_KEY:
    print("HATA: NVIDIA_API_KEY veya LOCAL_EMBEDDING_API_KEY env var gerekli!")
    print("  export NVIDIA_API_KEY=nvapi-XXXXX")
    sys.exit(1)

os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LOCAL_EMBEDDING_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["LOCAL_EMBEDDING_API_KEY"] = NVIDIA_API_KEY
os.environ["LOCAL_EMBEDDING_INPUT_TYPE"] = "auto"

# Repo path - klonlanmış repo root'undan çalıştır
REPO_PATH = os.environ.get("YARGI_MCP_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_PATH)

from semantic_search.embedder import (
    LocalEmbedder,
    is_semantic_search_available,
    is_local_embedding_configured,
)
from semantic_search.vector_store import VectorStore
from semantic_search.processor import DocumentProcessor

# Türkçe hukuki test verisi
QUERY = "Mirasçının muvazaalı satış işlemine karşı tapu iptali ve tescil davası açması"

DOCUMENTS = [
    "Davacı, murisi tarafından mirasından mal kaçırmak amacıyla üçüncü kişiye satışı yapılan tapuda kayıtlı taşınmazın iptali ve tesciline karar verilmesini talep etmektedir.",
    "Muvazaalı işlem nedeniyle tapu iptal ve tescil davasında süre, mirasçının işlemin muvazaalı olduğunu öğrendiği tarihten itibaren başlar.",
    "İş sözleşmesinin feshinde kıdem tazminatı hesaplanırken çalışanın kıdemi ve son brüt ücreti esas alınır.",
    "Ceza davasında sanığın haksız tahriki davranışları, hükmün açıklanmasının geri bırakılmasına engel teşkil etmez.",
    "Kamulaştırma bedelinin artırılması davasında zamanaşımı süresi kamulaştırma tarihinden itibaren hesaplanır.",
    "Muris muvazaası nedeniyle açılan tapu iptal ve tescil davasında, davanın kabulü halinde satış işleminin iptaline ve mirasçılar adına tesciline karar verilir.",
]
DOC_TITLES = [
    "Yargıtay 1. Hukuk Dairesi",
    "Yargıtay 2. Hukuk Dairesi",
    "Yargıtay 9. Hukuk Dairesi",
    "Yargıtay 4. Ceza Dairesi",
    "Danıştay 6. Dairesi",
    "Yargıtay Hukuk Genel Kurulu",
]

# Test edilecek NVIDIA modelleri
MODELS = [
    ("nvidia/nv-embed-v1", 4096, "raw"),         # Genel amaçlı NV-Embed
    ("nvidia/nv-embedqa-e5-v5", 1024, "raw"),    # E5 tabanlı multilingual (önerilen)
]

def test_model(model_name: str, dimension: int, prompt_style: str):
    print(f"\n{'='*70}")
    print(f"NVIDIA Model: {model_name}")
    print(f"Boyut: {dimension}, prompt_style: {prompt_style}")
    print(f"{'='*70}")

    # Override env vars
    os.environ["LOCAL_EMBEDDING_MODEL"] = model_name
    os.environ["LOCAL_EMBEDDING_DIMENSION"] = str(dimension)
    os.environ["EMBEDDING_PROMPT_STYLE"] = prompt_style

    # Konfigürasyon kontrolü
    assert is_local_embedding_configured(), "EMBEDDING_PROVIDER=local ayarlı değil"
    assert is_semantic_search_available(), "Semantik arama aktif değil"

    # Embedder oluştur
    embedder = LocalEmbedder()
    print(f"  ✓ Embedder oluşturuldu: {embedder.model}, dim={embedder.dimension}")
    print(f"  ✓ Asimetrik model mi? {embedder._is_asymmetric_model()}")

    # VectorStore + Processor (mcp_server_main.py ile aynı kurulum)
    vector_store = VectorStore(dimension=embedder.dimension)
    processor = DocumentProcessor(chunk_size=1500, chunk_overlap=300)

    # 1. Query embedding
    t0 = time.time()
    query_emb = embedder.encode_query(QUERY, task="search result")
    query_time = round(time.time() - t0, 3)
    print(f"  ✓ Query embedding: shape={query_emb.shape}, {query_time}s")

    # 2. Document processing + embedding
    docs_data = []
    for i, (doc_text, title) in enumerate(zip(DOCUMENTS, DOC_TITLES)):
        chunks = processor.process_document(
            document_id=f"doc_{i}",
            text=doc_text,
            metadata={"title": title, "doc_index": i},
        )
        if chunks:
            full_text = " ".join([c.text for c in chunks])
            docs_data.append({
                "id": f"doc_{i}",
                "text": full_text[:3000],
                "title": title,
                "metadata": {"title": title, "doc_index": i},
            })

    t0 = time.time()
    doc_texts = [d["text"] for d in docs_data]
    doc_titles = [d["title"] for d in docs_data]
    doc_embs = embedder.encode_documents(doc_texts, titles=doc_titles)
    docs_time = round(time.time() - t0, 3)
    print(f"  ✓ {len(docs_data)} doküman embeddingi: shape={doc_embs.shape}, {docs_time}s")

    # 3. VectorStore'a ekle
    vector_store.add_documents(
        ids=[d["id"] for d in docs_data],
        texts=doc_texts,
        embeddings=doc_embs,
        metadata=[d["metadata"] for d in docs_data],
    )

    # 4. Search (mcp_server_main.py ile aynı parametreler: top_k=10, threshold=0.3)
    results = vector_store.search(query_embedding=query_emb, top_k=10, threshold=0.3)

    # 5. Format & display (mcp_server_main.py'deki gibi)
    print(f"\n  Semantik Arama Sonuçları (top_k=10, threshold=0.3):")
    print(f"  {'Sıra':<5} {'Skor':<10} {'Doküman':<35} {'Önizleme':<60}")
    print(f"  {'-'*5} {'-'*10} {'-'*35} {'-'*60}")
    formatted = []
    for rank, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title", "?")
        idx = doc.metadata.get("doc_index", -1)
        marker = "★" if idx in (0, 1, 5) else " "  # Muvazaa-tapu iptali ile ilgili olanlar
        preview = doc.text[:55].replace("\n", " ")
        print(f"  {rank:<5} {score:<10.4f} {marker} {title:<32} {preview:<60}")
        formatted.append({
            "rank": rank,
            "doc_index": idx,
            "title": title,
            "score": float(score),
            "preview": doc.text[:200],
        })

    # Stats
    stats = vector_store.get_stats()
    print(f"\n  VectorStore stats: {stats['num_documents']} docs, "
          f"{stats['memory_usage_mb']:.3f} MB")

    return {
        "model": model_name,
        "dimension": dimension,
        "asymmetric": embedder._is_asymmetric_model(),
        "query_time_sec": query_time,
        "documents_time_sec": docs_time,
        "results": formatted,
        "top_match": formatted[0] if formatted else None,
    }


def main():
    print("NVIDIA + yargi-mcp LocalEmbedder Entegrasyon Testi")
    print(f"  Query: {QUERY}")
    print(f"  Toplam doküman: {len(DOCUMENTS)} (3 tanesi muvazaa/tapu ile ilgili: #0, #1, #5)")

    all_results = []
    for model_name, dim, ps in MODELS:
        try:
            r = test_model(model_name, dim, ps)
            all_results.append(r)
        except Exception as e:
            import traceback
            print(f"\n✗ {model_name} hatası: {e}")
            traceback.print_exc()
            all_results.append({
                "model": model_name,
                "error": str(e),
            })

    # Özet
    print(f"\n\n{'='*70}")
    print("ÖZET KARŞILAŞTIRMA")
    print(f"{'='*70}")
    print(f"{'Model':<32} {'Asim.':<7} {'Top-1':<8} {'Top-2':<8} {'Top-3':<8} {'Türkçe':<8}")
    print(f"{'-'*32} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<32} {'ERR':<7} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
            continue
        top_indices = [x["doc_index"] for x in r["results"][:3]]
        top1 = top_indices[0] if len(top_indices) > 0 else -1
        top2 = top_indices[1] if len(top_indices) > 1 else -1
        top3 = top_indices[2] if len(top_indices) > 2 else -1
        # Türkçe başarısı: top-3 içinde 2+ muvazaa dokümanı var mı
        muvazaa_in_top3 = sum(1 for i in top_indices if i in (0, 1, 5))
        tr_ok = "✓" if muvazaa_in_top3 >= 2 else "✗"
        asim = "✓" if r.get("asymmetric") else "✗"
        print(f"{r['model']:<32} {asim:<7} #{top1:<7} #{top2:<7} #{top3:<7} {tr_ok:<8}")

    # JSON kaydet
    out_path = "/home/z/my-project/scripts/nvidia_local_embedder_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetaylı sonuçlar: {out_path}")


if __name__ == "__main__":
    main()
