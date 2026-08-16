# Changelog

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenecektir.

Format [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) standardına uyar ve bu proje
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) sürümleme takip eder.

## [Unreleased]

### Planlanan
- `yargi-cli` TypeScript port'a semantik arama desteği (şu anda sadece yargi-mcp Python'da)
- ChromaDB kalıcı vector store entegrasyonu (şu anda in-memory)
- Streaming semantic search (chunk-by-chunk embed)
- Karar metni ön-işleme pipeline (HTML→Markdown→chunk + Türkçe stop-word removal)

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
