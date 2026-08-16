"""
Groq LLM hız testi — v1.3.0'da multi-provider LLM client'ın Groq desteğini doğrular.

Amaç:
  - Kullanıcının verdiği Groq API key ile gerçek LLM çağrısı yap
  - NVIDIA (60-240s) ile Groq (~500 tok/s) hız farkını ölç
  - Türkçe hukuki cevap kalitesini doğrula

Çalıştırma:
  python3 /home/z/my-project/scripts/test_groq_llm.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Repo path
REPO = Path("/home/z/my-project/repos/yargi-mcp-nvidia-semantic")
sys.path.insert(0, str(REPO))

# Groq API key — env var'dan oku (hardcoded değil)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# NVIDIA API key (önceki session'dan, env var'dan oku)
NVIDIA_API_KEY = os.environ.get("LOCAL_EMBEDDING_API_KEY", "")

# Test soruları
TEST_QUESTIONS = [
    "Mirasçı, murisin muvazaalı satış yaptığı taşınmaz için hangi davayı açmalıdır?",
    "Muvazaalı işlemlerde ispat yükü kimdedir?",
    "Tapu iptal ve tescil davasında görevli mahkeme hangisidir?",
]


def discover_groq_models(client):
    """Groq'ta kullanılabilir modelleri listele."""
    print("\n=== Groq modelleri listeleniyor ===")
    try:
        models = client.models.list()
        chat_models = []
        for m in models.data:
            # Sadece chat-completion modelleri
            if "llama" in m.id.lower() or "gemma" in m.id.lower() or "mixtral" in m.id.lower() or "qwen" in m.id.lower():
                ctx = getattr(m, "context_window", "?")
                chat_models.append((m.id, ctx))
                print(f"  • {m.id}  (ctx={ctx})")
        return chat_models
    except Exception as e:
        print(f"  HATA: {e}")
        return []


def test_groq_speed(model_id: str, api_key: str):
    """Groq ile 3 hukuki soru test et, hız + kalite ölç."""
    print(f"\n=== Groq Test: {model_id} ===")
    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    results = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n--- Q{i}: {q} ---")
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Sen Türk hukuku konusunda uzman bir asistansın. Cevaplarını Türkçe ver ve mümkün olduğunca kısa tut."},
                    {"role": "user", "content": q},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            elapsed = time.time() - t0
            answer = resp.choices[0].message.content or ""
            usage = resp.usage
            tokens = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

            print(f"  Süre: {elapsed:.2f}s")
            print(f"  Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={tokens}")
            print(f"  Hız: {tok_per_sec:.1f} tok/s")
            print(f"  Cevap (ilk 200 char): {answer[:200]}")

            results.append({
                "question": q,
                "model": model_id,
                "elapsed_s": round(elapsed, 2),
                "tokens": tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": round(tok_per_sec, 1),
                "answer": answer,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  HATA ({elapsed:.2f}s): {e}")
            results.append({
                "question": q,
                "model": model_id,
                "elapsed_s": round(elapsed, 2),
                "error": str(e),
            })

    return results


def main():
    print("=" * 80)
    print("GROQ LLM HIZ TESTİ — v1.4.0")
    print("=" * 80)
    print(f"Repo: {REPO}")
    print(f"Python: {sys.executable}")

    from openai import OpenAI

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )

    # 1. Modelleri listele
    available = discover_groq_models(client)
    if not available:
        print("\n!!! Hiç model bulunamadı, çıkılıyor")
        sys.exit(1)

    # 2. Test edilecek modeller — en yaygın olanları seç
    test_models = []
    # Öncelik sırası: llama-3.3-70b > llama-3.1-8b > gemma2-9b
    priority_ids = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
    ]
    available_ids = [m[0] for m in available]
    for pid in priority_ids:
        if pid in available_ids:
            test_models.append(pid)

    if not test_models:
        # İlk bulduğumuz 2 modeli test et
        test_models = [m[0] for m in available[:2]]

    print(f"\nTest edilecek modeller: {test_models}")

    all_results = {}
    for model_id in test_models:
        results = test_groq_speed(model_id, GROQ_API_KEY)
        all_results[model_id] = results

    # 3. Sonuçları kaydet
    out_file = REPO / "tests" / "v14_groq_llm_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "test": "Groq LLM speed test for v1.4.0",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models_tested": test_models,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n=== Sonuçlar kaydedildi: {out_file} ===")

    # 4. Özet tablo
    print("\n" + "=" * 80)
    print("ÖZET — Groq LLM Hız Testi")
    print("=" * 80)
    print(f"{'Model':<32} {'Soru':<3} {'Süre(s)':<10} {'Tokens':<8} {'tok/s':<10}")
    print("-" * 80)
    for model, results in all_results.items():
        for i, r in enumerate(results, 1):
            elapsed = r.get("elapsed_s", "?")
            tokens = r.get("tokens", "?")
            tps = r.get("tokens_per_sec", "?")
            err = r.get("error")
            if err:
                print(f"{model:<32} {i:<3} HATA: {err[:50]}")
            else:
                print(f"{model:<32} {i:<3} {elapsed:<10} {tokens:<8} {tps:<10}")

    # 5. NVIDIA ile karşılaştırma (reference)
    print("\n" + "=" * 80)
    print("NVIDIA REFERANSI (v1.3.0 worklog'undan):")
    print("  - meta/llama-3.1-70b-instruct: 46-87s/sorgu, ~50 tok/s")
    print("  - tokens/sorgu: ~2000-4000")
    print("Groq'ta beklenti: <10s/sorgu, ~500-1000 tok/s")


if __name__ == "__main__":
    main()
