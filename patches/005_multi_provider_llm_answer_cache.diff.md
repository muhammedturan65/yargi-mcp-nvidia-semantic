# Patch 005: Multi-Provider LLM + Semantik Answer Cache (v1.3.0)

**Tarih:** 2026-08-17
**Sürüm:** v1.2.0 → v1.3.0
**Dosyalar:** `qa_rag/llm_client.py`, `qa_rag/answer_cache.py` (yeni), `qa_rag/rag_engine.py`, `qa_rag/__init__.py`, `pyproject.toml`, `.env.example`

## Problem

v1.2.0'da RAG pipeline çalışıyor ama 3 kritik sorun var:

1. **NVIDIA LLM 60-240 saniye latency** — kullanıcının her soruya 1-4 dk beklemesi gerekiyor
2. **Tek provider (NVIDIA)** — alternatif LLM yok, kullanıcı Groq/OpenAI/Ollama kullanamaz
3. **Cache yok** — aynı soru tekrar sorulduğunda LLM tekrar çağrılıyor, tokens boşa harcanıyor

## Çözüm

### 1. Multi-Provider LLM Client (`qa_rag/llm_client.py`)

Tüm provider'lar (NVIDIA, Groq, OpenAI, Ollama) OpenAI-compatible API sunduğu için
tek SDK + tek sınıf (`LLMClient`) ile 4 provider desteklenir.

**Önce:**
```python
from qa_rag.llm_client import NvidiaLLMClient
client = NvidiaLLMClient()  # Sadece NVIDIA
```

**Sonra:**
```python
from qa_rag.llm_client import LLMClient
client = LLMClient()                       # LLM_PROVIDER env'den (default: nvidia)
client = LLMClient(provider="groq")        # Explicit Groq
# Backward compat:
client = NvidiaLLMClient()                 # Hâlâ çalışır (LLMClient alias)
```

### 2. Semantik Answer Cache (`qa_rag/answer_cache.py` — yeni)

ChromaDB'de ayrı `qa_cache` collection'ı. Query embedding ile cache'de cosine search,
threshold ≥ 0.92 ise HIT (cache'den cevap), değilse MISS (LLM çağrısı + cache'e yaz).

### 3. RAG Engine Entegrasyonu (`qa_rag/rag_engine.py`)

`ask()` metoduna cache lookup + store eklendi. `RAGContext.query_embedding` alanı eklendi
(cache lookup için NVIDIA embed'i tekrar kullanır).

## Before/After

### Hız Karşılaştırması

| Senaryo | v1.2.0 | v1.3.0 cache MISS | v1.3.0 cache HIT |
|---|---|---|---|
| Retrieval | 1.5s | 5.8s | 1.85s |
| LLM call | 46-240s | 46.6s | **0s (atlandı)** |
| Cache lookup | — | 0s | 4ms |
| **Toplam** | **60-240s** | **52.5s** | **1.86s** |
| Hızlanma | — | — | **28.3x** |

### Provider Karşılaştırması

| Provider | Default Model | Hız | Maliyet | Türkçe |
|---|---|---|---|---|
| nvidia | meta/llama-3.1-70b-instruct | Yavaş (~5 tok/s) | Ücretsiz | İyi |
| **groq** | llama-3.3-70b-versatile | **Hızlı (~500 tok/s)** | Ücretsiz | İyi |
| openai | gpt-4o-mini | Orta (~50 tok/s) | Ucuz | Çok iyi |
| ollama | llama3.1:8b | Local | Ücretsiz | Orta |

### Kod Değişiklikleri

| Dosya | Değişiklik |
|---|---|
| `qa_rag/llm_client.py` | Refactor: `LLMClient` (multi-provider) + `NvidiaLLMClient` alias |
| `qa_rag/answer_cache.py` | **Yeni** (295 satır) — ChromaDB-backed semantik cache |
| `qa_rag/rag_engine.py` | +`llm_provider`, `enable_answer_cache` parametreleri; cache entegrasyonu |
| `qa_rag/__init__.py` | +`LLMClient`, `AnswerCache`, `CacheHit`, `get_llm_client` exports |
| `pyproject.toml` | v1.2.0 → v1.3.0, keywords'e groq/openai/ollama/answer-cache |
| `.env.example` | +`LLM_PROVIDER`, `LLM_API_KEY`, `RAG_ANSWER_CACHE`, `RAG_CACHE_THRESHOLD` |
| `CHANGELOG.md` | v1.3.0 bölümü eklendi |
| `README.md` | v1.3.0 bölümü + mimari diyagramı eklendi |

## Yeni Env Var'lar

```bash
# Multi-provider LLM
LLM_PROVIDER=nvidia          # nvidia|groq|openai|ollama
LLM_API_KEY=...              # Seçili provider'ın key'i
LLM_MODEL=...                # Override (default provider'dan gelir)
LLM_TIMEOUT=90               # Saniye
LLM_TEMPERATURE=0.2          # Hukuki: düşük yaratıcılık
LLM_MAX_TOKENS=1500

# Provider-specific keys (LLM_PROVIDER seçimine göre)
NVIDIA_API_KEY=nvapi-...
GROQ_API_KEY=gq-...
OPENAI_API_KEY=sk-...
OLLAMA_API_KEY=ollama        # Ollama key istemez

# Semantik answer cache
RAG_ANSWER_CACHE=true        # Default açık
RAG_CACHE_THRESHOLD=0.92     # Cosine threshold
RAG_CACHE_COLLECTION=qa_cache
```

## Backward Compatibility

v1.3.0, v1.2.0 ve v1.1.0 kodu ile **tam uyumlu**:

- ✅ `NvidiaLLMClient` alias korundu (v1.1.0/v1.2.0 import'ları çalışır)
- ✅ Tüm eski env var'lar destekleniyor: `NVIDIA_API_KEY`, `NVIDIA_LLM_MODEL`, vb.
- ✅ `LegalQARAG()` default hâlâ NVIDIA + cache enabled
- ✅ v1.2.0 ChromaDB collection'ları (yargi_decisions, yargi_v12_medium) çalışır
- ✅ Eski `RAGResponse` API'si korundu, yeni alanlar opsiyonel default değerlerle

## Test Sonuçları

### Smoke Test (5/5 PASS)

```
✓ Import testleri
✓ LLMClient factory (4 provider, config resolve, ollama no-key)
✓ AnswerCache init (ChromaDB collection oluşturma)
✓ AnswerCache store+lookup (3 Q→A, exact HIT score=1.0, unrelated MISS)
✓ LegalQARAG init (yeni parametreler + lazy cache)
```

### RAG Cache Benchmark

```
Soru: "Muvazaalı tapu satışında mirasçı hangi davayı açar?"

Cache MISS (ilk sorgu):
  - Retrieval: 5800ms (NVIDIA query embed + ChromaDB search)
  - LLM call: 46641ms (NVIDIA Llama 3.1 70B, 3808 tokens)
  - Cache store: ~50ms
  - Toplam: 52.5s

Cache HIT (ikinci sorgu):
  - Retrieval: 1853ms (NVIDIA query embed + ChromaDB search)
  - Cache lookup: 4ms (ChromaDB cosine search)
  - LLM call: 0ms (ATLANDI)
  - Toplam: 1.86s

Hızlanma: 28.3x
Cache score: 1.0000 (exact match)
```

Test dosyaları:
- `/home/z/my-project/scripts/test_v13_smoke.py`
- `/home/z/my-project/scripts/test_v13_cache_check.py`
- `tests/v13_rag_cache_results.json`

## Bilinen Sınırlar

1. **NVIDIA query embedding hâlâ her sorguda yapılır** (~1s)
   - v1.4.0'da query embedding cache eklenecek (NVIDIA API çağrısı tamamen önlenebilir)
2. **Cache TTL yok**
   - Hukuki soruların cevabı değişmez, karar metinleri sabit olduğu için sorun değil
3. **Cache invalidation manuel**
   - Corpus güncellendiğinde `cache.clear()` çağrısı gerekir
4. **Groq free tier limiti**
   - 30 req/dk, 14400 req/gün — production için paid plan önerilir
