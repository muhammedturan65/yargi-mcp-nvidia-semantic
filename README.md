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

## Hukuki QA Chatbot (RAG) — v1.1.0

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

