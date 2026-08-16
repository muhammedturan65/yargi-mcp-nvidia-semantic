#!/usr/bin/env python3
"""
NVIDIA API'sinin embedding modellerini test eder.
Türkçe hukuki metinlerle kıyaslama yapar.

Çalıştırma:
    python /home/z/my-project/scripts/test_nvidia_embeddings.py
"""
import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any

# NVIDIA API anahtarı — env var'dan al (hardcoded değil)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LOCAL_EMBEDDING_API_KEY")
if not NVIDIA_API_KEY:
    print("HATA: NVIDIA_API_KEY veya LOCAL_EMBEDDING_API_KEY env var gerekli!")
    print("  export NVIDIA_API_KEY=nvapi-XXXXX")
    sys.exit(1)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Test edilecek modeller (Türkçe için potansiyel adaylar)
MODELS_TO_TEST = [
    "nvidia/nv-embedqa-e5-v5",       # E5 tabanlı, multilingual (en iyi Türkçe adayı)
    "nvidia/nv-embed-v1",             # NV-Embed v1 (çalıştığı doğrulandı)
    "nvidia/nv-embedqa-mistral-7b-v2", # Mistral tabanlı
    "nvidia/llama-nemotron-embed-1b-v2", # Yeni Nemotron
    "nvidia/embed-qa-4",              # Embed QA 4
    "snowflake/arctic-embed-l",       # Snowflake (yeniden dene)
]

# Türkçe hukuki test metinleri (hukuk alanından)
QUERY = "Mirasçının muvazaalı satış işlemine karşı tapu iptali ve tescil davası açması"

DOCUMENTS = [
    "Davacı, murisi tarafından mirasından mal kaçırmak amacıyla üçüncü kişiye satışı yapılan tapuda kayıtlı taşınmazın iptali ve tesciline karar verilmesini talep etmektedir.",
    "Muvazaalı işlem nedeniyle tapu iptal ve tescil davasında süre, mirasçının işlemin muvazaalı olduğunu öğrendiği tarihten itibaren başlar.",
    "İş sözleşmesinin feshinde kıdem tazminatı hesaplanırken çalışanın kıdemi ve son brüt ücreti esas alınır.",
    "Ceza davasında sanığın haksız tahriki davranışları, hükmün açıklanmasının geri bırakılmasına engel teşkil etmez.",
    "Kamulaştırma bedelinin artırılması davasında zamanaşımı süresi kamulaştırma tarihinden itibaren hesaplanır.",
    "Muris muvazaası nedeniyle açılan tapu iptal ve tescil davasında, davanın kabulü halinde satış işleminin iptaline ve mirasçılar adına tesciline karar verilir.",
]

def test_embedding_model(model_name: str) -> Dict[str, Any]:
    """Belirli bir NVIDIA embedding modelini test eder."""
    print(f"\n{'='*60}")
    print(f"Test ediliyor: {model_name}")
    print(f"{'='*60}")

    try:
        from openai import OpenAI
    except ImportError:
        print("openai paketi gerekli. Yükleniyor...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "openai"])
        from openai import OpenAI

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY,
    )

    result = {
        "model": model_name,
        "status": "unknown",
        "dimension": None,
        "error": None,
        "query_time_sec": None,
        "documents_time_sec": None,
        "similarities": [],
        "top_match": None,
    }

    # NV-Embed modelleri için input_type kullanılıyor (özel olarak)
    # E5-based NVIDIA models (nv-embedqa-e5-v5) asimetrik: input_type zorunlu
    is_e5 = "e5" in model_name.lower() and "nv-embed" not in model_name.lower()
    is_nv_embedqa_e5 = "nv-embedqa-e5" in model_name.lower()
    is_nv_embed = "nv-embed" in model_name.lower() and "embedqa" not in model_name.lower()

    # --- 1. Query embedding ---
    try:
        query_text = f"query: {QUERY}" if is_e5 else QUERY
        t0 = time.time()
        # nv-embedqa-e5-v5 için input_type query, diğer NV modeller için de query
        # (asimetrik modeller için zorunlu)
        extra_body = {}
        if is_nv_embedqa_e5 or is_nv_embed:
            extra_body["input_type"] = "query"
        query_resp = client.embeddings.create(
            model=model_name,
            input=[query_text],
            encoding_format="float",
            extra_body=extra_body,
        )
        result["query_time_sec"] = round(time.time() - t0, 3)
        query_emb = np.array(query_resp.data[0].embedding, dtype=np.float32)
        result["dimension"] = len(query_emb)
        print(f"  ✓ Query embedding başarılı: {result['dimension']} boyutlu, {result['query_time_sec']}s")
    except Exception as e:
        result["status"] = "query_error"
        result["error"] = f"Query hatası: {type(e).__name__}: {e}"
        print(f"  ✗ {result['error']}")
        return result

    # --- 2. Document embeddings ---
    try:
        doc_texts = [f"passage: {d}" if is_e5 else d for d in DOCUMENTS]
        t0 = time.time()
        extra_body = {}
        if is_nv_embedqa_e5 or is_nv_embed:
            extra_body["input_type"] = "passage"
        doc_resp = client.embeddings.create(
            model=model_name,
            input=doc_texts,
            encoding_format="float",
            extra_body=extra_body,
        )
        result["documents_time_sec"] = round(time.time() - t0, 3)
        doc_embs = np.array(
            [d.embedding for d in sorted(doc_resp.data, key=lambda x: x.index)],
            dtype=np.float32,
        )
        print(f"  ✓ {len(DOCUMENTS)} doküman embeddingi: {doc_embs.shape}, {result['documents_time_sec']}s")
    except Exception as e:
        result["status"] = "doc_error"
        result["error"] = f"Doküman hatası: {type(e).__name__}: {e}"
        print(f"  ✗ {result['error']}")
        return result

    # --- 3. L2 normalize + cosine similarity ---
    query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    doc_embs = doc_embs / (norms + 1e-8)
    sims = doc_embs @ query_emb

    # En alakalıdan en az alakalıya doğru sırala
    ranked_idx = np.argsort(sims)[::-1]
    for idx in ranked_idx:
        result["similarities"].append({
            "doc_index": int(idx),
            "score": float(sims[idx]),
            "preview": DOCUMENTS[idx][:100],
        })

    result["top_match"] = result["similarities"][0]
    result["status"] = "success"

    print(f"\n  Sıralama (en alakalı → en az alakalı):")
    for s in result["similarities"]:
        marker = "★" if s["doc_index"] in (0, 5) else " "  # 0 ve 5: muvazaa ile ilgili
        print(f"    {marker} [{s['score']:.4f}] Doc #{s['doc_index']}: {s['preview'][:80]}...")

    # Türkçe hukuk için başarı kriteri: muvazaalı tapu iptali ile ilgili iki doküman (0 ve 5) top-2'de olmalı
    top2 = {result["similarities"][0]["doc_index"], result["similarities"][1]["doc_index"]}
    if top2 == {0, 5}:
        print(f"\n  ✓ Türkçe hukuki anlama BAŞARILI: top-2 doğru (Doc #0 ve #5)")
    else:
        print(f"\n  ⚠ Türkçe hukuki anlama ZAYIF: top-2 = {top2} (beklenen: {{0, 5}})")

    return result


def main():
    print("NVIDIA API Embedding Model Testi")
    print(f"Query: {QUERY[:80]}...")
    print(f"Toplam doküman: {len(DOCUMENTS)}")
    print(f"Test edilecek model sayısı: {len(MODELS_TO_TEST)}")

    results = []
    for model in MODELS_TO_TEST:
        try:
            r = test_embedding_model(model)
            results.append(r)
        except KeyboardInterrupt:
            print("İptal edildi.")
            break
        except Exception as e:
            print(f"Model {model} beklenmeyen hata: {e}")
            results.append({
                "model": model,
                "status": "fatal_error",
                "error": str(e),
            })

    # Özet rapor
    print(f"\n\n{'='*70}")
    print("ÖZET RAPOR")
    print(f"{'='*70}")
    print(f"{'Model':<45} {'Boyut':<8} {'Top-1 Score':<12} {'Türkçe':<8}")
    print(f"{'-'*45} {'-'*8} {'-'*12} {'-'*8}")
    for r in results:
        if r["status"] == "success":
            top1 = r["top_match"]["score"]
            top2_idx = {r["similarities"][0]["doc_index"], r["similarities"][1]["doc_index"]}
            tr_ok = "✓" if top2_idx == {0, 5} else "✗"
            print(f"{r['model']:<45} {r['dimension']:<8} {top1:<12.4f} {tr_ok:<8}")
        else:
            print(f"{r['model']:<45} {'ERR':<8} {'N/A':<12} {'N/A':<8}")

    # JSON kaydet
    out_path = "/home/z/my-project/scripts/nvidia_embeddings_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetaylı sonuçlar: {out_path}")

    # En iyi modeli seç ve öner
    successes = [r for r in results if r["status"] == "success"]
    if successes:
        # Türkçe kalitesini ve skoru birleştirerek değerlendir
        def score(r):
            top2_idx = {r["similarities"][0]["doc_index"], r["similarities"][1]["doc_index"]}
            tr_bonus = 0.2 if top2_idx == {0, 5} else 0
            return r["top_match"]["score"] + tr_bonus
        best = max(successes, key=score)
        print(f"\n★ ÖNERİLEN MODEL: {best['model']}")
        print(f"  Boyut: {best['dimension']}")
        print(f"  Top-1 skoru: {best['top_match']['score']:.4f}")
        print(f"  Query süresi: {best['query_time_sec']}s")
        print(f"  {len(DOCUMENTS)} doküman için: {best['documents_time_sec']}s")


if __name__ == "__main__":
    main()
