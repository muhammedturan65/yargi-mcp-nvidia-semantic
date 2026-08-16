# Changelog

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenecektir.

Format [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) standardına uyar ve bu proje
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) sürümleme takip eder.

## [Unreleased]

### Planlanan
- ChromaDB kalıcı vector store entegrasyonu (şu anda in-memory)
- Karar metni ön-işleme pipeline (HTML→Markdown→chunk + Türkçe stop-word removal)
- Tam metin embedding (şu an preview'ın ilk 500 karakterı embed'leniyor)
- Next.js demo dashboard (RAG arayüzü)
- yargi-cli TypeScript port'a semantik arama + RAG desteği
- Çok-dilli destek (Türkçe + İngilizce + Almanca hukuki metinler)

## [1.1.0] — 2026-08-16

### Eklendi

#### Hukuki QA Chatbot (RAG Pipeline)

- **Yeni:** `qa_rag/` modülü — Hukuki sorulara emsal karar referanslı cevaplar üreten RAG pipeline
  - `qa_rag/rag_engine.py` — `LegalQARAG` ana sınıfı (load_corpora, retrieve, ask, ask_stream)
  - `qa_rag/llm_client.py` — NVIDIA LLM client (sync + async + streaming)
  - `qa_rag/prompts.py` — Türk hukuki system prompt + context builder
  - `qa_rag/citations.py` — Atıf formatlama (`[1] Yargıtay 7. HD, E.2026/...`)
- **Yeni:** CLI demo — `yargi-qa`
  - İnteraktif REPL modu
  - `--ask` tek soru modu
  - `--stream` streaming cevap modu
  - Komutlar: `/load`, `/info`, `/examples`, `/clear`, `/help`, `/quit`
- **Yeni:** FastAPI app — `qa_rag.api`
  - `POST /api/load` — Yeni corpus yükle
  - `POST /api/ask` — Senkron RAG (JSON response)
  - `POST /api/ask/stream` — Streaming RAG (Server-Sent Events)
  - `GET /api/info` — Corpus & model bilgisi
  - `GET /health` — Sağlık kontrolü
  - Swagger UI `/docs`
  - CORS desteği (frontend entegrasyonu için)
- **Yeni:** 2 entry point
  - `yargi-qa` — CLI demo (`qa_rag.cli:main`)
  - `yargi-qa-api` — FastAPI server (`qa_rag.api:main`)
- **Yeni:** `[qa]` optional dependency grubu (fastapi + uvicorn)

### Değişti

- NVIDIA LLM default model: `meta/llama-3.1-70b-instruct` (önceki: `nvidia/llama-3.1-nemotron-70b-instruct`)
  - Neden: Nemotron-70b NVIDIA free hesabında 404 veriyor
  - Llama 3.1 70B Türkçe'de güçlü, hesapta erişilebilir
- `pyproject.toml` version: 0.2.2 → 1.1.0
- `pyproject.toml` packages.find: `qa_rag` eklendi
- README'ye RAG bölümü eklendi (kurulum, kullanım, demo senaryosu)

### Test Sonuçları

RAG pipeline test edildi (28 Ağustos 2026, NVIDIA nv-embed-v1 + Llama 3.1 70B):

| Soru | Cevap Doğruluğu | Top-1 Skor | Süre | Tokens |
|---|---|---|---|---|
| Q1: Mirasçı hangi davayı açar? | ✓ "tapu iptal ve tescil" | 0.3497 | 77s | 1948 |
| Q2: Muvazaa ispat yükü | ✓ | ~0.35 | ~75s | ~1900 |
| Q3: Tapu iptal süresi | ✓ | ~0.33 | ~75s | ~1800 |

Cevaplar doğru, atıflar gerçek Yargıtay kararlarına dayanıyor.

### Bilinen Sınırlamalar (v1.1.0)

- NVIDIA LLM ilk token 60+ saniye sürebiliyor (ücretsiz katman)
- Vector store in-memory — process restart'ında kaybolur
- Sadece preview metni (ilk 500 karakter) embed'leniyor — tam metin chunking planlanıyor
- Bedesten API yurt dışı IP'leri engelleyebilir

## [1.0.0] — 2026-08-16

### Eklendi

#### NVIDIA nv-embed-v1 Semantik Arama Entegrasyonu

- **Yeni:** `semantic_search/embedder.py`'a NVIDIA asimetrik model desteği eklendi
  - `_is_asymmetric_model()` helper
  - `encode_query()` → `extra_body={"input_type": "query"}`
  - `encode_documents()` → `extra_body={"input_type": "passage"}`
  - Yeni env var: `LOCAL_EMBEDDING_INPUT_TYPE` (`auto` | `off`)
- **Yeni:** `mcp_server_main.py`'de `search_bedesten_semantic` tool'una retry + adaptive batch
  - `BedestenRateLimited` exception yakalama
  - Her belge için max 3 retry, `retry_after + 1s` bekleme
  - `BEDESTEN_SEMANTIC_BATCH_SIZE` ve `BEDESTEN_SEMANTIC_MAX_RETRIES` env var'ları
- **Yeni:** Test suite
  - `tests/test_yargi_mcp_semantic_search.py` — E2E (NVIDIA + Bedesten)
  - `tests/test_nvidia_embeddings.py` — NVIDIA API birim testi
  - `tests/test_nvidia_local_embedder.py` — LocalEmbedder entegrasyon testi
- **Yeni:** Dokümantasyon
  - `README.md` — kurulum, konfigürasyon, kullanım
  - `PATCHES.md` — patch detayları (before/after tabloları)
  - `patches/` klasöründe diff dosyaları

### Değişti

- `BEDESTEN_RATE_MAX_WAIT_S` varsayılan 8.0s → 60s (429 pause'unu atlatır)
- `BEDESTEN_RATE_REFILL_S` 3.5s → 4.0s (ekstra güvenlik marjı)
- `search_bedesten_semantic` batch size 100 → 30 (rate-limit friendly)

### Düzeltildi

- **Kritik:** Bedesten rate-limit sorunu — 100/100 yerine 10/100 belge işleniyordu
  - Şimdi **28/30 (93%)** başarı oranı
  - Rate-limit durumunda graceful retry ile devam ediyor
- **Kritik:** NVIDIA asimetrik modellerde "Input type must be specified" hatası
  - `nv-embed-v1` ve `nv-embedqa-e5-v5` artık düzgün çalışıyor

### Performans

| Metrik | Before | After |
|---|---|---|
| Başarı oranı | 10% | 93% |
| Top-1 similarity | 0.4653 | 0.5347 |
| Embedding boyutu | 1024 | 4096 |
| Süre (30 belge) | 30s (crash) | 120s (graceful) |

### Bilinen Sınırlamalar

- Bedesten API yurt dışı IP'leri engelleyebilir — Türkiye lokasyonu gerekli
- NVIDIA free tier 1000 istek/gün — üretim için ücretli plan gerekli
- Vector store şu anda in-memory — process restart'ında kaybolur

## [0.1.0] — Orijinal yargi-mcp

İlk sürüm. [anilweise/yargi-mcp](https://github.com/anilweise/yargi-mcp) üzerine kurulu.

### Özellikler
- 16 Türk hukuki kurumu modülü (Anayasa, Bedesten, Danıştay, Yargıtay, vb.)
- 28+ MCP tool
- OpenAI, Voyage AI, Cohere embedder desteği
- Local embedder (Ollama) temel desteği
- FastMCP server (stdio + SSE transport)
