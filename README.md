# yargi-mcp + NVIDIA nv-embed-v1 Semantik Arama Entegrasyonu

Bu repo, [anilweise/yargi-mcp](https://github.com/anilweise/yargi-mcp) üzerine uygulanan **NVIDIA nv-embed-v1 semantik arama** ve **Bedesten rate-limit dayanıklılık** patch'lerini içerir. Tüm değişiklikler referans patches klasöründe diff dosyası olarak da mevcuttur.

## Özet

| Metrik | Orijinal | Patched |
|---|---|---|
| Hedef belge sayısı | 100 | 30 |
| Başarılı fetch | 10 / 100 (10%) | **28 / 30 (93%)** |
| Failed fetches | 90 | 2 |
| Toplam süre | ~30 saniye (rate-limit crash) | ~2 dakika (graceful) |
| Top-1 similarity skoru | 0.4653 | **0.5347** |
| Embedding boyutu | 1024 (nv-embedqa-e5-v5) | **4096** (nv-embed-v1) |

## İçerik

```
yargi-mcp-nvidia-semantic/
├── README.md                          # Bu dosya
├── PATCHES.md                         # Patch detayları (Seçenek 1 + 3)
├── CHANGELOG.md                       # Sürüm notları
├── LICENSE                            # Orijinal lisans (MIT)
├── pyproject.toml                     # Bağımlılıklar
├── .env.example                       # Çevre değişkenleri şablonu
├── Dockerfile
├── mcp_server_main.py                 # Patched: adaptive batch + retry
├── semantic_search/
│   ├── embedder.py                    # Patched: NVIDIA asimetrik model desteği
│   ├── vector_store.py
│   └── processor.py
├── bedesten_mcp_module/
│   └── client.py                      # BedestenRateLimited exception + rate limiter
├── *_mcp_module/                      # 16 Türk hukuki kurumu modülü
│   ├── anayasa_mcp_module/
│   ├── bedesten_mcp_module/
│   ├── danistay_mcp_module/
│   ├── emsal_mcp_module/
│   ├── gib_mcp_module/
│   ├── kik_mcp_module/
│   ├── kvkk_mcp_module/
│   ├── rekabet_mcp_module/
│   ├── sayistay_mcp_module/
│   ├── sigorta_tahkim_mcp_module/
│   └── ...
├── patches/                           # Diff dosyaları
│   ├── 001_embedder_nvidia_asymmetric_support.diff
│   └── 002_mcp_server_adaptive_batch_retry.diff
└── tests/
    ├── test_yargi_mcp_semantic_search.py    # E2E test (NVIDIA + Bedesten)
    ├── test_nvidia_embeddings.py            # NVIDIA API birim testi
    ├── test_nvidia_local_embedder.py        # LocalEmbedder entegrasyon testi
    └── yargi_mcp_semantic_results.json      # Gerçek test çıktısı (28 belge)
```

## Patch'ler

### Patch 1: NVIDIA Asimetrik Model Desteği (`semantic_search/embedder.py`)

NVIDIA `nv-embed-v1` ve `nv-embedqa-e5-v5` modelleri **asimetrik**'tir — query ve passage için farklı `input_type` gerektirir. Orijinal `LocalEmbedder` bunu desteklemiyordu.

**Çözüm:**
- `_is_asymmetric_model()` — model adından asimetrik tespiti
- `encode_query()` → `extra_body={"input_type": "query"}`
- `encode_documents()` → `extra_body={"input_type": "passage"}`
- Yeni env var: `LOCAL_EMBEDDING_INPUT_TYPE` (`auto` | `off`)

### Patch 2: Bedesten Rate-Limit Dayanıklılık (`mcp_server_main.py`)

Bedesten API 10 istek / 30 saniye pencere ile sınırlıdır. Orijinal kod 100 belgeyi sıralı çekmeye çalışıyor, 10. istekten sonra rate-limit'e takılıp kalan 90 belge atlanıyordu.

**Çözüm (Seçenek 1 + 3 Birleşik):**

1. **Adaptive batch** — `all_decisions[:100]` → `all_decisions[:BEDESTEN_SEMANTIC_BATCH_SIZE]` (varsayılan 30)
2. **Retry loop** — her belge için `BedestenRateLimited` yakalanır, `retry_after + 1s` bekleyip max 3 kez retry yapılır
3. **Toleranslı max_wait** — `BEDESTEN_RATE_MAX_WAIT_S` 8s → 60s'e yükseltildi (429 pause'unu atlatır)
4. **Yavaş refill** — `BEDESTEN_RATE_REFILL_S` 3.5s → 4.0s (ekstra güvenlik marjı)

Tüm parametreler env var ile kontrol edilebilir.

## Kurulum

```bash
# Repoyu klonla
git clone https://github.com/muhammedturan65/yargi-mcp-nvidia-semantic.git
cd yargi-mcp-nvidia-semantic

# Python >= 3.11 gerekli
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Konfigürasyon

### NVIDIA API Anahtarı

[ NVIDIA build.nvidia.com](https://build.nvidia.com) üzerinden ücretsiz API anahtarı al. Asimetrik model desteği için `nv-embed-v1` önerilir (4096 boyut, daha uzun context).

### Çevre Değişkenleri

```bash
# === NVIDIA Embedding ===
export EMBEDDING_PROVIDER=local
export LOCAL_EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1
export LOCAL_EMBEDDING_API_KEY=nvapi-XXXXXXXX
export LOCAL_EMBEDDING_MODEL=nvidia/nv-embed-v1
export LOCAL_EMBEDDING_DIMENSION=4096
export LOCAL_EMBEDDING_INPUT_TYPE=auto
export EMBEDDING_PROMPT_STYLE=raw

# === Bedesten Rate-Limit (Patch 2) ===
export BEDESTEN_RATE_CAPACITY=1
export BEDESTEN_RATE_REFILL_S=4.0
export BEDESTEN_RATE_MAX_WAIT_S=60
export BEDESTEN_SEMANTIC_BATCH_SIZE=30
export BEDESTEN_SEMANTIC_MAX_RETRIES=3
```

## Test

```bash
# E2E test (NVIDIA + Bedesten gerçek API çağrısı)
python tests/test_yargi_mcp_semantic_search.py

# NVIDIA API birim testi
python tests/test_nvidia_embeddings.py

# LocalEmbedder entegrasyon testi
python tests/test_nvidia_local_embedder.py
```

### Beklenen Çıktı

```
[1/4] mcp_server_main yükleniyor...
  SEMANTIC_SEARCH_AVAILABLE = True
  ✓ BedestenApiClient hazır: BedestenApiClient

[2/4] search_bedesten_semantic tool çağrılıyor...

[3/4] Sonuç alındı!
  status: success
  total_documents_processed: 28
  embedding_model: nvidia/nv-embed-v1
  embedding_dimension: 4096
  documents_in_store: 28
  failed_fetches: 2
```

## MCP Client Konfigürasyonu

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "yargi-mcp": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "yargi_mcp"],
      "env": {
        "EMBEDDING_PROVIDER": "local",
        "LOCAL_EMBEDDING_BASE_URL": "https://integrate.api.nvidia.com/v1",
        "LOCAL_EMBEDDING_API_KEY": "nvapi-...",
        "LOCAL_EMBEDDING_MODEL": "nvidia/nv-embed-v1",
        "LOCAL_EMBEDDING_DIMENSION": "4096",
        "LOCAL_EMBEDDING_INPUT_TYPE": "auto",
        "BEDESTEN_RATE_MAX_WAIT_S": "60",
        "BEDESTEN_SEMANTIC_BATCH_SIZE": "30"
      }
    }
  }
}
```

### 5ire MCP Client

5ire ayarlarında `5ire-settings.png` dosyasına bakın — aynı env var'lar orada da kullanılır.

## Lisans

Orijinal yargi-mcp projesi MIT lisansı altında dağıtılmaktadır. Bu fork da aynı lisansı korur.

## Katkıda Bulunanlar

- **Orijinal yargi-mcp**: [anilweise](https://github.com/anilweise)
- **NVIDIA + rate-limit patch'leri**: [muhammedturan65](https://github.com/muhammedturan65)

## İlgili Belgeler

- `PATCHES.md` — Patch detayları (before/after tabloları, env var referansı)
- `CHANGELOG.md` — Sürüm notları
- `CLAUDE.md` — Claude Code için geliştirici notları (orijinal repo)

---

## Hukuki QA Chatbot (RAG) — v1.3.0 (Multi-Provider LLM + Answer Cache)

v1.3.0, v1.2.0'daki en büyük acı noktasını çözer: **NVIDIA LLM 60-240 saniye latency**.
İki katmanlı çözüm ile tekrarlayan/benzer sorular artık ~2 saniyede döner.

### Yenilikler

#### 1. Multi-Provider LLM Backend

Artık NVIDIA'ya bağlı kalmak zorunda değilsiniz. **Groq** ile ~500 tok/s, **OpenAI** ile
ucuz/hızlı GPT-4o-mini, **Ollama** ile tamamen local LLM kullanabilirsiniz.

```bash
# Hızlı LLM için (önerilen):
export LLM_PROVIDER=groq
export GROQ_API_KEY=gq_...

# Veya OpenAI:
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...

# Veya tamamen local:
export LLM_PROVIDER=ollama
ollama pull llama3.1:8b
```

| Provider | Default Model | Hız | Maliyet | Türkçe Kalite |
|---|---|---|---|---|
| nvidia | meta/llama-3.1-70b-instruct | Yavaş (~5 tok/s) | Ücretsiz | İyi |
| **groq** | llama-3.3-70b-versatile | **Hızlı (~500 tok/s)** | Ücretsiz tier | İyi |
| openai | gpt-4o-mini | Orta (~50 tok/s) | Ucuz | Çok iyi |
| ollama | llama3.1:8b | Local hız | Ücretsiz | Orta |

#### 2. Semantik Answer Cache

Aynı veya benzer (cosine ≥ 0.92) soru tekrar sorulduğunda, LLM çağrısı yapılmadan
cache'den yanıt döner. ChromaDB'de ayrı `qa_cache` collection'da saklanır.

```python
from qa_rag import LegalQARAG

rag = LegalQARAG()
await rag.load_corpora()

# İlk sorgu — LLM çağrılır (~60s NVIDIA, ~3s Groq)
r1 = await rag.ask("Mirasçı hangi davayı açar?")
print(f"Süre: {r1.total_time_ms/1000:.1f}s, from_cache: {r1.from_cache}")
# → Süre: 52.5s, from_cache: False

# Aynı soru tekrar — cache HIT, LLM atlanır
r2 = await rag.ask("Mirasçı hangi davayı açar?")
print(f"Süre: {r2.total_time_ms/1000:.1f}s, from_cache: {r2.from_cache}, score: {r2.cache_score:.4f}")
# → Süre: 1.9s, from_cache: True, score: 1.0000
```

Cache kontrolü:
```bash
export RAG_ANSWER_CACHE=true             # default — açık
export RAG_CACHE_THRESHOLD=0.92          # cosine threshold (yüksek = daha sıkı eşleşme)
export RAG_CACHE_COLLECTION=qa_cache     # ChromaDB collection adı
```

### v1.2.0 → v1.3.0 Performans Karşılaştırması

| Senaryo | v1.2.0 | v1.3.0 cache MISS | v1.3.0 cache HIT |
|---|---|---|---|
| İlk sorgu | 60-240s | 52.5s | — |
| Tekrar sorgu | 60-240s (LLM yine çağrılır) | — | **1.86s** |
| LLM tokens | ~3800 | ~3800 | 0 (atlandı) |
| **Hızlanma** | — | — | **28.3x** |

### Mimari (v1.3.0)

```
                     ┌─────────────────────────────────────┐
                     │  BedestenIndexer (tek seferlik)     │
                     │  v1.2.0 — ChromaDB kalıcı store    │
                     └─────────────────┬───────────────────┘
                                       │ (disk - kalıcı)
                                       ▼
                     ┌─────────────────────────────────────┐
Kullanıcı sorusu ──► │  LegalQARAG.ask()                   │
                     │   1. NVIDIA nv-embed-v1 (query)     │ ~1s
                     │   2. ChromaDB.search_with_dedup()   │ ~50ms
                     │   3. AnswerCache.lookup() [v1.3.0]  │ ~4ms
                     │      ├─ HIT → cache'den cevap       │ → RETURN
                     │      └─ MISS → devam                │
                     │   4. LLMClient.chat_async() [v1.3.0]│ ~3-240s
                     │      └─ NVIDIA / Groq / OpenAI /    │
                     │         Ollama (env'den seçim)      │
                     │   5. AnswerCache.store() [v1.3.0]   │ ~50ms
                     └─────────────────────────────────────┘
```

### Backward Compatibility

v1.3.0, v1.2.0 ve v1.1.0 kodu ile **tam uyumlu**:
- `NvidiaLLMClient` alias korundu (v1.1.0/v1.2.0 import'ları çalışır)
- Tüm eski env var'lar destekleniyor: `NVIDIA_API_KEY`, `NVIDIA_LLM_MODEL`, vb.
- `LegalQARAG()` default hâlâ NVIDIA + cache enabled
- v1.2.0 ChromaDB collection'ları (yargi_decisions, yargi_v12_medium) çalışır

### Hızlı Başlangıç (v1.3.0 — Groq ile)

```bash
# 1. Groq API key al: https://console.groq.com/keys (ücretsiz)
export GROQ_API_KEY=gq_...

# 2. NVIDIA embedding (hâlâ NVIDIA nv-embed-v1 kullanıyoruz — en iyi Türkçe embedding)
export LOCAL_EMBEDDING_API_KEY=nvapi-...
export EMBEDDING_PROVIDER=local
export LOCAL_EMBEDDING_BASE_URL=https://integrate.api.nvidia.com/v1
export LOCAL_EMBEDDING_MODEL=nvidia/nv-embed-v1
export LOCAL_EMBEDDING_DIMENSION=4096
export LOCAL_EMBEDDING_INPUT_TYPE=auto
export EMBEDDING_PROMPT_STYLE=raw

# 3. Provider seç
export LLM_PROVIDER=groq

# 4. Cache açık (default)
export RAG_ANSWER_CACHE=true

# 5. İlk seferlik index (~5-10 dk, ChromaDB'ye yazılır)
yargi-qa --load-corpus

# 6. Sor
yargi-qa --ask "Muvazaalı tapu satışında mirasçı hangi davayı açar?"
# → İlk sorgu: ~3s (Groq LLM)
# → Tekrar: ~1.9s (cache HIT)
```

### v1.3.0 Test Sonuçları

- **5/5 smoke test** geçti (import, factory, cache init, store+lookup, RAG init)
- **Cache HIT benchmark**: 52.5s → 1.86s (28.3x speedup, score=1.0000)
- Test dosyaları:
  - `tests/v13_rag_cache_results.json` — benchmark sonuçları
  - 5/5 smoke test, 1/1 RAG benchmark

---

## Hukuki QA Chatbot (RAG) — v1.2.0 (ChromaDB Kalıcı Store)

v1.2.0, RAG pipeline'ına **ChromaDB kalıcı vector store** + **token-aware chunking** ekler. Bir kez indexlenen kararlar process restart'ında kaybolmaz, sorgular sub-second hızda döner.

### Mimari

```
                     ┌─────────────────────────────────────┐
                     │  BedestenIndexer (tek seferlik)     │
                     │                                     │
Bedesten API ──────► │  search → fetch full text → chunk   │
                     │     ↓                               │
                     │  NVIDIA nv-embed-v1 (passage)       │
                     │     ↓                               │
                     │  ChromaDB.add_chunks()              │
                     └─────────────────┬───────────────────┘
                                       │ (disk - kalıcı)
                                       ▼
                     ┌─────────────────────────────────────┐
Kullanıcı sorusu ──► │  LegalQARAG.ask()                   │
                     │   1. NVIDIA nv-embed-v1 (query)     │ ~1s
                     │   2. ChromaDB.search_with_dedup()   │ ~50ms
                     │   3. build_context_from_decisions() │
                     │   4. NVIDIA Llama 3.1 70B           │ ~60-240s
                     └─────────────────┬───────────────────┘
                                       ▼
                            Atıflı cevap: "tapu iptali ve tescil davası açar.
                            [1] Yargıtay 7. HD, E.2026/2403, K.2026/3418"
```

### Modül Yapısı (v1.2.0)

```
qa_rag/
├── __init__.py        # Modül girişi + exports
├── llm_client.py      # NVIDIA LLM client (sync + async + streaming)
├── prompts.py         # Türk hukuki system prompt + context builder
├── citations.py       # Atıf formatlama
├── rag_engine.py      # LegalQARAG (chroma/memory backend seçimi)
├── chunker.py         # LegalChunker — 512-token, section-aware
├── indexer.py         # BedestenIndexer — Bedesten → ChromaDB pipeline
├── cli.py             # yargi-qa interaktif REPL + --ask modu
└── api.py             # FastAPI app (REST + SSE streaming)

semantic_search/
├── embedder.py        # NVIDIA nv-embed-v1 embedder (query/passage asimetrik)
├── vector_store.py    # In-memory VectorStore (v1.0.0'dan)
└── vector_store_chroma.py  # ChromaDB kalıcı store (v1.2.0)
```

### v1.1.0 → v1.2.0 İyileştirmeler

| Metrik | v1.1.0 | v1.2.0 |
|---|---|---|
| Retrieval süresi | ~120s (her sorguda Bedesten fetch) | ~1s (ChromaDB'den okuma) |
| Top-1 similarity | 0.35 | **0.53** (+51%) |
| Kalıcılık | Yok (in-memory) | **Var** (ChromaDB disk) |
| Tam metin embedding | Yok (500 char preview) | **Var** (512-token chunk) |
| Process restart | Corpus kaybolur | **Korunur** |
| NVIDIA API çağrısı/sorgu | 30+ (her belge) | **1** (sadece query) |

### Kurulum (v1.2.0)

```bash
# ChromaDB + tiktoken otomatik kurulur (pyproject.toml dependencies)
cd yargi-mcp-nvidia-semantic
pip install -e ".[qa]"

# NVIDIA API key
export NVIDIA_API_KEY=nvapi-...

# ChromaDB kalıcı dizini (default: ./chroma_db)
export CHROMA_PERSIST_DIR=/path/to/chroma_db
```

### Kullanım — İlk Index (tek seferlik, ~5-10 dk)

```python
import asyncio
from qa_rag import LegalQARAG

async def main():
    rag = LegalQARAG(backend="chroma")
    # İlk sefer: 200 belge çek, chunk'la, embed'le, ChromaDB'ye yaz
    result = await rag.load_corpora(
        initial_keyword="muvazaa tapu iptal",
        court_types=["YARGITAYKARARI"],
        target_docs=200,
    )
    print(f"{result['indexed_docs']} belge, {result['total_chunks']} chunk indexlendi")
    print(f"Süre: {result['elapsed_s']}s")

asyncio.run(main())
```

### Kullanım — Sorgu (sub-second retrieval)

```python
rag = LegalQARAG(backend="chroma")
# ChromaDB'de veri varsa is_corpora_loaded=True (process restart sonrası bile)
print(rag.is_corpora_loaded)  # True

response = await rag.ask("Mirasçı muvazaalı satışa karşı hangi davayı açar?")
print(response.answer)
print(f"Retrieval: {response.retrieval_time_ms}ms")  # ~1000ms
print(f"LLM: {response.generation_time_ms}ms")       # ~60-240s
```

### Çevre Değişkenleri (v1.2.0)

#### ChromaDB
- `CHROMA_PERSIST_DIR` — Kalıcı dizin (default: `./chroma_db`)
- `CHROMA_COLLECTION` — Collection adı (default: `yargi_decisions`)
- `CHROMA_DISTANCE` — Distance metric: `cosine`|`l2`|`ip` (default: `cosine`)

#### Indexer
- `INDEXER_BATCH_SIZE` — NVIDIA'ya bir seferde kaç chunk embed (default: 32)
- `INDEXER_TARGET_DOCS` — Hedef belge sayısı (default: 200)
- `INDEXER_KEYWORDS` — Virgülle ayrılmış anahtar kelimeler
- `INDEXER_COURT_TYPES` — Virgülle ayrılmış mahkeme tipleri

### Bilinen Sınırlar

1. **NVIDIA LLM yavaş** — İlk token 60-240 saniye. Streaming modu UX'i iyileştirir.
2. **Bedesten rate-limit** — 10 istek/30s, indexleme süresini sınırlar (~5 dk/50 belge).
3. **Query embedding cache yok** — Her sorgu NVIDIA'ya gider (NVIDIA embed ~1s).
4. **Bedesten yurt dışı IP'leri engelleyebilir** — Türkiye lokasyonu gerekli.

### Demo Senaryosu

```bash
# 1. İlk index (tek seferlik, ~5-10 dk)
export NVIDIA_API_KEY=nvapi-...
export CHROMA_PERSIST_DIR=$HOME/yargi_chroma
python -c "
import asyncio
from qa_rag import LegalQARAG
async def m():
    rag = LegalQARAG(backend='chroma')
    r = await rag.load_corpora(target_docs=50)
    print(r)
asyncio.run(m())
"

# 2. API başlat (artık ChromaDB'de veri var, anında hazır)
yargi-qa-api &

# 3. Soru sor (sub-second retrieval)
curl -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Muvazaalı satışta mirasçının hakları nelerdir?"}'
```

---

## Hukuki QA Chatbot (RAG) — v1.1.0 (legacy, in-memory)

Bu sürüm, semantik arama pipeline'ının üzerine bir **RAG tabanlı hukuki asistan** ekler. Kullanıcı doğal dilde soru sorar, sistem en alakalı emsal kararları bulur ve NVIDIA LLM ile atıflı cevap üretir.

### Mimari

```
Kullanıcı sorusu
       ↓
[NVIDIA nv-embed-v1] — query embedding (4096d)
       ↓
[VectorStore.search] — top-K en alakalı kararlar (cosine similarity)
       ↓
[build_context_from_decisions] — numaralandırılmış karar metinleri
       ↓
[NVIDIA Llama 3.1 70B Instruct] — atıflı cevap üretimi
       ↓
Atıf listesi: [1] Yargıtay 7. HD, E.2026/2403, K.2026/3418
```

### Modül Yapısı

```
qa_rag/
├── __init__.py        # Modül girişi
├── llm_client.py      # NVIDIA LLM client (sync + async + streaming)
├── prompts.py         # Türk hukuki system prompt + context builder
├── citations.py       # Atıf formatlama
├── rag_engine.py      # LegalQARAG ana sınıf
├── cli.py             # yargi-qa interaktif REPL + --ask modu
└── api.py             # FastAPI app (REST + SSE streaming)
```

### Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/muhammedturan65/yargi-mcp-nvidia-semantic.git
cd yargi-mcp-nvidia-semantic

# Python >= 3.11 ile venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[qa]"

# NVIDIA API key set et
export NVIDIA_API_KEY=nvapi-XXXXXXXX
```

### Kullanım — CLI (yargi-qa)

#### İnteraktif REPL

```bash
yargi-qa
```

İlk çalıştırmada Bedesten'den 30 karar çeker (~2 dk), sonraki sorular hızlı çalışır.

```
========================================================================
  yargi-qa — Türk Hukuki QA Chatbot (RAG)
  NVIDIA nv-embed-v1 + Llama 3.1 70B + Bedesten emsal kararları
========================================================================

İlk kurulum: Bedesten'den 30 karar yükleniyor (~2 dk)...
  keyword: 'muvazaa tapu iptal'

✓ 28 karar yüklendi (2 fetch hatası)

Komutlar:
  /load <keyword>   Yeni corpus yükle (örn: /load nafaka)
  /info             Mevcut corpus bilgisi
  /examples         Örnek hukuki soruları göster
  /clear            Ekranı temizle
  /help             Bu yardım
  /quit             Çıkış

yargi-qa> Mirasçı, murisin muvazaalı satışına karşı hangi davayı açar?
Düşünüyor...

------------------------------------------------------------------------
Soru: Mirasçı, murisin muvazaalı satışına karşı hangi davayı açar?
------------------------------------------------------------------------

Mirasçı, murisin muvazaalı satışına karşı tapu iptal ve tescil davası açabilir.
...

[1] Yargıtay 7. HD, E.2026/2403, K.2026/3418 (14.03.2002) [skor: 0.3497]
[2] Yargıtay 7. HD, E.2026/4216, K.2026/3514 (29.06.2026) [skor: 0.3422]
[3] Yargıtay 7. HD, E.2026/3025, K.2026/3499 (17.10.2019) [skor: 0.3268]

⏱  Toplam: 76800ms (retrieval: 1003ms, LLM: 75770ms) | Tokens: 1948
------------------------------------------------------------------------
```

#### Tek Soru Modu

```bash
yargi-qa --ask "Muvazaa iddiasında ispat yükü kimin üzerinedir?"
```

#### Streaming Modu (token token)

```bash
yargi-qa --ask "Tapu iptal davasında süre nedir?" --stream
```

### Kullanım — FastAPI (yargi-qa-api)

```bash
# API server başlat
yargi-qa-api
# veya
uvicorn qa_rag.api:app --host 0.0.0.0 --port 8001
```

#### Endpoint'ler

| Method | Endpoint | Açıklama |
|---|---|---|
| `POST` | `/api/load` | Yeni corpus yükle (Bedesten'den) |
| `POST` | `/api/ask` | Senkron RAG (JSON response) |
| `POST` | `/api/ask/stream` | Streaming RAG (SSE) |
| `GET` | `/api/info` | Corpus & model bilgisi |
| `GET` | `/health` | Sağlık kontrolü |
| `GET` | `/docs` | Swagger UI |

#### Örnek İstek

```bash
# Corpus yükle (önce bunu yap)
curl -X POST http://localhost:8001/api/load \
  -H "Content-Type: application/json" \
  -d '{"initial_keyword": "muvazaa tapu iptal"}'

# Soru sor
curl -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Mirasçı muvazaalı satışa karşı hangi davayı açar?",
    "top_k": 5,
    "temperature": 0.2
  }'

# Streaming
curl -N -X POST http://localhost:8001/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Tapu iptal davasında süre nedir?"}'
```

### Test Sonuçları (28 Ağustos 2026)

| Soru | Cevap Doğruluğu | Top-1 Skor | Süre | Tokens |
|---|---|---|---|---|
| Q1: Mirasçı hangi davayı açar? | ✓ "tapu iptal ve tescil" | 0.3497 | 77s | 1948 |
| Q2: Muvazaa ispat yükü | ✓ | 0.35 | ~75s | ~1900 |
| Q3: Tapu iptal süresi | ✓ | 0.33 | ~75s | ~1800 |

Detaylı JSON: `tests/qa_rag_test_results.json`

### Çevre Değişkenleri

#### NVIDIA LLM

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `NVIDIA_API_KEY` | (zorunlu) | build.nvidia.com API key |
| `NVIDIA_LLM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA API URL |
| `NVIDIA_LLM_MODEL` | `meta/llama-3.1-70b-instruct` | LLM model adı |
| `NVIDIA_LLM_TEMPERATURE` | `0.2` | Düşük = belirleyici (hukuki için) |
| `NVIDIA_LLM_MAX_TOKENS` | `1500` | Maksimum cevap uzunluğu |
| `NVIDIA_LLM_TIMEOUT` | `90` | Saniye |

**NVIDIA Hesabında Doğrulanmış Modeller:**
- ✓ `meta/llama-3.1-70b-instruct` (önerilen, Türkçe güçlü)
- ✓ `meta/llama-3.1-8b-instruct` (hızlı alternatif)
- ✗ `nvidia/llama-3.1-nemotron-70b-instruct` (hesapta 404)
- ✗ `meta/llama-3.1-405b-instruct` (hesapta 404)

#### RAG Pipeline

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `QA_AUTO_LOAD_CORPUS` | `0` | `1` = API başlangıcında otomatik corpus yükle |
| `BEDESTEN_SEMANTIC_BATCH_SIZE` | `30` | `load_corpora` batch boyutu |
| `QA_TOP_K` | `5` | LLM'e kaç karar feed'lenecek |

### Bilinen Sınırlamalar

1. **Vector store in-memory** — Process restart'ında kaybolur. Gelecek sürümde ChromaDB entegrasyonu planlanıyor.
2. **NVIDIA LLM yavaş** — İlk token bazen 60+ saniye. Streaming modu UX'i iyileştirir.
3. **Bedesten yurt dışı IP'leri engelleyebilir** — Türkiye lokasyonu gerekli.
4. **Preview metin limiti** — Şu an karar metninin ilk 500 karakterı embed'leniyor. İleride tam metin + chunking planlanıyor.

### Demo Senaryosu

```bash
# 1. API başlat
export NVIDIA_API_KEY=nvapi-...
export QA_AUTO_LOAD_CORPUS=1
yargi-qa-api &

# 2. Bekle (corpus yükleniyor, ~2 dk)
sleep 120

# 3. Soru sor
curl -X POST http://localhost:8001/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Muvazaalı satışta mirasçının hakları nelerdir?"}'

# 4. Streaming dene
curl -N -X POST http://localhost:8001/api/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Tapu iptal davasında karar süresi nedir?"}'
```

