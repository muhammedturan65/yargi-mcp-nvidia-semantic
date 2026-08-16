# Changelog

Bu projedeki tüm önemli değişiklikler bu dosyada belgelenecektir.

Format [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) standardına uyar ve bu proje
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) sürümleme takip eder.

## [Unreleased]

### Planlanan
- Next.js demo dashboard (RAG arayüzü)
- yargi-cli TypeScript port'a semantik arama + RAG desteği
- Çok-dilli destek (Türkçe + İngilizce + Almanca hukuki metinler)
- Query embedding cache (sorgu→embedding lookup, NVIDIA API çağrısını azaltır)
- Hukuki stop-word filtering + section-aware retrieval (GEREKÇE section'ına ağırlık)

## [1.3.0] — 2026-08-17

### Eklendi

#### Multi-Provider LLM Backend + Semantik Answer Cache

v1.3.0, v1.2.0'daki en büyük kullanıcı acısını çözer: **NVIDIA LLM 60-240 saniye latency**.
İki katmanlı çözüm uygulanmıştır:

1. **Multi-provider LLM client** — NVIDIA/Groq/OpenAI/Ollama destekli
2. **Semantik answer cache** — ChromaDB'de Q→A çiftleri, tekrarlayan/benzer sorular anında döner

##### 1. Multi-Provider LLM (`qa_rag/llm_client.py`)

- **Yeni:** `LLMClient` sınıfı — tüm provider'lar OpenAI-compatible API sunduğu için tek SDK
  - 4 provider desteği: `nvidia` (default) / `groq` (önerilen — ~500 tok/s) / `openai` / `ollama`
  - Provider seçimi: `LLM_PROVIDER` env var veya constructor parametresi
  - Provider-specific config: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_TIMEOUT`
  - Backward compat: `NvidiaLLMClient` alias korundu (v1.1.0/v1.2.0 kodu çalışır)
- **Yeni:** `get_llm_client()` factory fonksiyonu
- **Yeni:** `PROVIDER_DEFAULTS` dict — her provider için default base_url/model/timeout
- **Yeni:** `LLMResponse.provider` alanı — hangi provider'ın ürettiğini belirtir
- **Yeni:** `LLMResponse.from_cache` alanı — cache hit mi LLM call mı

Provider default'ları:
| Provider | Base URL | Default Model | Timeout | Notes |
|---|---|---|---|---|
| nvidia | integrate.api.nvidia.com/v1 | meta/llama-3.1-70b-instruct | 90s | Kaliteli, yavaş |
| groq | api.groq.com/openai/v1 | llama-3.3-70b-versatile | 30s | **HIZLI (~500 tok/s)** |
| openai | api.openai.com/v1 | gpt-4o-mini | 30s | Ucuz, hızlı |
| ollama | localhost:11434/v1 | llama3.1:8b | 60s | Local, API key yok |

Eski env var'lar (`NVIDIA_API_KEY`, `NVIDIA_LLM_MODEL`, vb.) hâlâ destekleniyor — mevcut
kurulumlar kırılmadan çalışmaya devam eder.

##### 2. Semantik Answer Cache (`qa_rag/answer_cache.py`)

- **Yeni:** `AnswerCache` sınıfı — ChromaDB'de ayrı `qa_cache` collection'ı
  - `lookup(question_embedding, threshold)` — cosine search, top-1, threshold kontrolü
  - `store(question, embedding, answer, citations, metadata)` — yeni Q→A çifti yaz
  - `clear()` — tüm cache'i temizle
  - `size()`, `get_stats()` — cache durumu
- **Yeni:** `CacheHit` dataclass — cache hit sonucu (answer, citations, score, metadata)
- **ChromaVectorStore ile aynı persistent client'ı paylaşır** — ayrı process/DB gerekmez

Çalışma prensibi:
```
ask(question)
  ├─ retrieve(question)          # NVIDIA embed (~1s) + ChromaDB search (~50ms)
  ├─ cache.lookup(query_emb)     # ChromaDB cosine search (~4ms)
  │   ├─ HIT (score >= 0.92)    # → cached answer + citations döner, LLM atlanır
  │   └─ MISS                   # → LLM çağrısı yapılır
  ├─ llm.chat_async(messages)    # NVIDIA: ~50-240s / Groq: ~2-5s
  └─ cache.store(Q, emb, A, cit) # Sonraki sorgu için cache'e yaz
```

Env var'lar:
- `RAG_ANSWER_CACHE` — `"true"` (default) / `"false"` (kapat)
- `RAG_CACHE_THRESHOLD` — `0.92` (default, cosine threshold)
- `RAG_CACHE_COLLECTION` — `"qa_cache"` (default, ChromaDB collection adı)

##### 3. RAG Engine Entegrasyonu (`qa_rag/rag_engine.py`)

- `LegalQARAG.__init__()` — yeni parametreler: `llm_provider`, `enable_answer_cache`
- `_get_llm_client()` — `NvidiaLLMClient` yerine `LLMClient` factory kullanır
- `_get_answer_cache()` — lazy-init answer cache
- `retrieve()` — `RAGContext.query_embedding` alanı eklendi (cache lookup için reuse)
- `ask()` — cache lookup + cache store entegrasyonu
- `RAGResponse` — yeni alanlar: `from_cache`, `cache_score`, `llm_provider`
- Cache HIT durumunda LLM çağrısı tamamen atlanır, `generation_time_ms` ~4ms

### Değişti

- `qa_rag/__init__.py` — `LLMClient`, `get_llm_client`, `AnswerCache`, `CacheHit`, `LLMResponse` export edildi
- `__version__` — `1.2.0` → `1.3.0`
- `pyproject.toml` — version + description + keywords güncellendi

### Performans Karşılaştırması

Benchmark: "Muvazaalı tapu satışında mirasçı hangi davayı açar?" sorusu, NVIDIA LLM
(meta/llama-3.1-70b-instruct), ChromaDB 70 chunk (17 belge) corpus.

| Senaryo | v1.2.0 | v1.3.0 (cache miss) | v1.3.0 (cache hit) |
|---|---|---|---|
| Retrieval | 1.5s | 5.8s (NVIDIA query embed) | 1.85s |
| LLM call | 46-240s | 46.6s (3808 tokens) | **0s (atlandı)** |
| Cache lookup | — | 0s (cache boş) | 4ms |
| **Toplam** | **60-240s** | **52.5s** | **1.86s** |
| Hızlanma | — | — | **28.3x** |

Not: Cache HIT durumunda hâlâ ~1.8s var çünkü NVIDIA query embedding hâlâ yapılıyor.
v1.4.0'da query embedding cache eklenecek (NVIDIA API çağrısı tamamen önlenebilir).

### Gerçek Kullanım Senaryoları

1. **Demo / sohbet**: Aynı soru 2-3 kere sorulduğunda, ilk seferden sonra anında cevap
2. **Toplu soru-cevap**: 100 belge üzerinde 20 farklı soru → ilk tur 17 dk, ikinci tur 36s
3. **Production**: Sık sorulan sorular (FAQ) için cache hit ratio %80+ ulaşabilir

### Test Sonuçları

- **5/5 smoke test** geçti: import, factory, cache init, store+lookup, RAG init
- **Cache HIT benchmark**: 28.3x speedup (52.5s → 1.86s, score=1.0000)
- Test dosyaları:
  - `/home/z/my-project/scripts/test_v13_smoke.py`
  - `/home/z/my-project/scripts/test_v13_cache_check.py`
  - `tests/v13_rag_cache_results.json` — benchmark sonuçları

### Bilinen Sınırlar

- NVIDIA query embedding hâlâ her sorguda yapılır (~1s) — v1.4.0'da cache eklenecek
- Cache TTL yok — hukuki soruların cevabı değişmez, karar metinleri sabit olduğu için sorun değil
- Cache invalidation: corpus güncellendiğinde manuel `cache.clear()` gerekir
- Groq free tier 30 req/dk limiti var (production için paid plan önerilir)

## [1.2.0] — 2026-08-17

### Eklendi

#### ChromaDB Kalıcı Vector Store + Token-Aware Chunking

- **Yeni:** `semantic_search/vector_store_chroma.py` — ChromaDB-backed kalıcı vector store
  - `ChromaVectorStore` sınıfı, mevcut `VectorStore` ile API uyumlu (drop-in replacement)
  - `add_documents()` — document-level index (vector_store.py ile uyumlu)
  - `add_chunks()` — chunk-level index (document_id metadata ile)
  - `search()` — cosine similarity search, ChromaDB HNSW index
  - `search_with_dedup()` — chunk-level arama + document bazında dedup
  - `list_documents_by_metadata()` — metadata'ya göre filtreleme (court_type, birim_adi, vb)
  - ChromaDB PersistentClient ile disk'e yazma — process restart'ında kayıp yok
  - Resume desteği: idempotent upsert + mevcut document_id'leri atlama
- **Yeni:** `qa_rag/chunker.py` — Token-aware hukuki metin chunker
  - `LegalChunker` sınıfı: 512-token hedef, 80-token overlap, section-aware
  - Türk hukuki karar yapısını tanır (GEREKÇE, HÜKÜM, ÖZET, KARAR bölümleri)
  - tiktoken (cl100k_base) ile gerçek token sayımı — Llama 3 tokenizer ile uyumlu
  - Cümle sınırlarında bölme (Türkçe kısaltmaları korur: Dr., Prof., Av.)
  - Çok küçük chunk'ları otomatik merge
- **Yeni:** `qa_rag/indexer.py` — Bedesten → ChromaDB index pipeline
  - `BedestenIndexer` sınıfı: tam otomatik index pipeline
  - Bedesten search → tam metin fetch → chunk → NVIDIA embed → ChromaDB yaz
  - Multi-keyword destek (virgülle ayrılmış liste)
  - Multi-court-type destek (YARGITAYKARARI, DANISTAYKARAR, vb)
  - Resume desteği: ChromaDB'de var olan belgeleri atla
  - Checkpoint: JSON'a ilerleme yaz (`last_index.json`)
  - Hata toleransı: tek belge hatası tüm pipeline'ı durdurmaz
- **Yeni:** `LegalQARAG` backend seçimi
  - `backend="chroma"` (default): Kalıcı ChromaDB store
  - `backend="memory"` (legacy): v1.1.0 davranışı, in-memory store
  - `chroma_collection` parametresi ile çoklu collection desteği

### Değişti

- **Kritik:** RAG retrieval artık kalıcı ChromaDB'den okuyor
  - **v1.1.0:** Her sorguda Bedesten fetch + embed + search (~2 dk/sorgu)
  - **v1.2.0:** ChromaDB'den sub-second retrieval (~1 saniye/sorgu)
  - **Hız artışı: ~120x** (NVIDIA query embedding dahil)
  - Process restart'ında corpus kaybolmuyor
- **Kritik:** Tam metin embedding (preview yerine)
  - **v1.1.0:** İlk 500 karakter embed'leniyordu (düşük kalite)
  - **v1.2.0:** Tam metin 512-token chunk'lara bölünür, her chunk ayrı embed'lenir
  - Retrieval kalitesi arttı: top-1 skor 0.35 → 0.53 (50% daha yüksek)
- `pyproject.toml` version: 1.1.0 → 1.2.0
- `pyproject.toml` dependencies: `chromadb>=0.5.0`, `tiktoken>=0.7.0` eklendi
- `qa_rag/__init__.py` yeni exports: `LegalChunker`, `Chunk`, `chunk_text`, `BedestenIndexer`, `IndexResult`, `IndexProgress`

### Test Sonuçları (v1.2.0)

#### Retrieval Benchmark (17 belge / 70 chunk, ChromaDB)

| Soru | Süre | Top-1 Skor | Sonuç Sayısı |
|---|---|---|---|
| Mirasçı muvazaalı satışa karşı hangi davayı açar? | 1354ms | 0.5275 | 5 |
| Muvazaa iddiasında ispat yükü kimdedir? | 1147ms | 0.4157 | 5 |
| Tapu iptal davası açma süresi nedir? | 1064ms | 0.5036 | 5 |
| Muris muvazaası nedir ve nasıl ispatlanır? | 395ms | 0.3899 | 5 |
| Tapu iptal ve tescil davasında görevli mahkeme | 1308ms | 0.6090 | 5 |

**Ortalama retrieval: 1054ms/sorgu** (NVIDIA query embed ~1000ms + ChromaDB search ~50ms)

#### Tam RAG Pipeline (LLM çağrısı dahil)

| Soru | Retrieval | LLM | Toplam | Tokens |
|---|---|---|---|---|
| Mirasçı hangi davayı açar? | 827ms | 240s | 241.8s | 3974 |
| Muvazaa ispat yükü | 561ms | 72s | 72.6s | 3816 |

Cevaplar doğru (tapu iptali ve tescil davası, tenkis davası, ispat yükü iddia sahibinde),
atıflar gerçek Yargıtay kararlarına dayanıyor (E.2026/2403, E.2025/5182, vb).

#### v1.1.0 vs v1.2.0 Karşılaştırma

| Metrik | v1.1.0 | v1.2.0 | İyileşme |
|---|---|---|---|
| Retrieval süresi | ~120s (Bedesten fetch) | ~1s (ChromaDB) | **120x hız** |
| Top-1 similarity | 0.35 | 0.53 | **51% daha yüksek** |
| Kalıcılık | Yok (in-memory) | Var (ChromaDB disk) | ✓ |
| Tam metin embedding | Yok (preview 500 char) | Var (512-token chunk) | ✓ |
| Process restart | Corpus kaybolur | Korunur | ✓ |
| NVIDIA API çağrısı/sorgu | 30+ (her belge) | 1 (sadece query) | **30x azalma** |

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

### Bilinen Sınırlamalar (v1.2.0)

- NVIDIA LLM ilk token 60-240 saniye sürebilir (ücretsiz katman oranı)
- Bedesten rate-limit (10 istek/30s) indexleme süresini sınırlar (~5 dk/50 belge)
- ChromaDB ilk açılışta ~2 saniye yüklenir (warm-up)
- Query embedding cache henüz yok — her sorgu NVIDIA'ya gider

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
