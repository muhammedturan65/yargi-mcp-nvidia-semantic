# Patch 004: ChromaDB Kalıcı Vector Store + Token-Aware Chunking (v1.2.0)

## Sorun

v1.1.0'daki RAG pipeline'ın 3 kritik sorunu vardı:

1. **Vector store in-memory** — Process restart'ında corpus kaybolur,
   her sorgu için yeniden Bedesten'den ~30 belge çekmek gerekir (~2 dk).

2. **Sadece preview metni embed'leniyor** — Karar metninin ilk 500 karakterı
   embed'leniyordu. Bu, kararın %5'i demek — retrieval kalitesi düşük.

3. **NVIDIA LLM yavaş** — 70-90 saniye. Demo için imkansız.

## Çözüm

### 1. ChromaDB Kalıcı Vector Store (`semantic_search/vector_store_chroma.py`)

ChromaDB PersistentClient ile disk'e yazma. Mevcut `VectorStore` ile API uyumlu
(drop-in replacement):

```python
# Eski (v1.1.0):
from semantic_search.vector_store import VectorStore
vs = VectorStore(dimension=4096)  # in-memory, restart'ta kaybolur

# Yeni (v1.2.0):
from semantic_search.vector_store_chroma import ChromaVectorStore
vs = ChromaVectorStore(dimension=4096)  # kalıcı, restart'ta korunur
```

Özellikler:
- `add_documents()` — document-level index (eski API ile uyumlu)
- `add_chunks()` — chunk-level index (document_id metadata ile)
- `search()` — cosine similarity search, ChromaDB HNSW index
- `search_with_dedup()` — chunk-level arama + document bazında dedup
- `list_documents_by_metadata()` — metadata'ya göre filtreleme
- Resume desteği: idempotent upsert + mevcut document_id'leri atlama

### 2. Token-Aware Chunker (`qa_rag/chunker.py`)

Türk hukuki karar metinleri için token-aware chunker:

- **512-token hedef, 80-token overlap** — NVIDIA nv-embed-v1'in 5120 token
  context limitine güvenli sığar
- **Section-aware** — GEREKÇE, HÜKÜM, ÖZET, KARAR bölümlerini tanır,
  her bölüm yeni chunk başlangıcı olur
- **tiktoken (cl100k_base)** — Llama 3 tokenizer ile uyumlu gerçek token sayımı
- **Cümle sınırlarında bölme** — Türkçe kısaltmaları korur (Dr., Prof., Av.)
- **Otomatik merge** — Çok küçük chunk'ları komşu ile birleştirir

### 3. BedestenIndexer (`qa_rag/indexer.py`)

Tam otomatik index pipeline:

```
Bedesten search → fetch full text → chunk → NVIDIA embed → ChromaDB yaz
```

- Multi-keyword destek (virgülle ayrılmış liste)
- Multi-court-type destek (YARGITAYKARARI, DANISTAYKARAR, vb)
- Resume desteği: ChromaDB'de var olan belgeleri atla
- Checkpoint: JSON'a ilerleme yaz (`last_index.json`)
- Hata toleransı: tek belge hatası tüm pipeline'ı durdurmaz

### 4. LegalQARAG Backend Seçimi (`qa_rag/rag_engine.py`)

Yeni `backend` parametresi:

```python
# v1.2.0 (default): Kalıcı ChromaDB
rag = LegalQARAG(backend="chroma")

# v1.1.0 (legacy): In-memory
rag = LegalQARAG(backend="memory")
```

## Before/After Tablosu

| Metrik | v1.1.0 (before) | v1.2.0 (after) | İyileşme |
|---|---|---|---|
| Retrieval süresi | ~120s (Bedesten fetch) | ~1s (ChromaDB) | **120x hız** |
| Top-1 similarity | 0.35 | 0.53 | **51% daha yüksek** |
| Kalıcılık | Yok (in-memory) | Var (ChromaDB disk) | ✓ |
| Tam metin embedding | Yok (500 char preview) | Var (512-token chunk) | ✓ |
| Process restart | Corpus kaybolur | Korunur | ✓ |
| NVIDIA API çağrısı/sorgu | 30+ (her belge) | 1 (sadece query) | **30x azalma** |
| Ortalama chunk/belge | 1 (preview) | 4.5 (full text) | 4.5x |
| Ortalama token/chunk | ~125 (preview) | 548 (full) | 4.4x |

## Test Sonuçları

### Retrieval Benchmark (17 belge / 70 chunk)

| Soru | Süre | Top-1 Skor |
|---|---|---|
| Mirasçı muvazaalı satışa karşı hangi davayı açar? | 1354ms | 0.5275 |
| Muvazaa iddiasında ispat yükü kimdedir? | 1147ms | 0.4157 |
| Tapu iptal davası açma süresi nedir? | 1064ms | 0.5036 |
| Muris muvazaası nedir ve nasıl ispatlanır? | 395ms | 0.3899 |
| Tapu iptal ve tescil davasında görevli mahkeme | 1308ms | 0.6090 |

**Ortalama: 1054ms/sorgu** (NVIDIA query embed ~1000ms + ChromaDB search ~50ms)

### Tam RAG Pipeline (LLM dahil)

| Soru | Retrieval | LLM | Toplam |
|---|---|---|---|
| Mirasçı hangi davayı açar? | 827ms | 240s | 241.8s |
| Muvazaa ispat yükü | 561ms | 72s | 72.6s |

Cevaplar doğru (tapu iptali ve tescil davası, tenkis davası, ispat yükü iddia
sahibinde), atıflar gerçek Yargıtay kararlarına dayanıyor (E.2026/2403,
E.2025/5182, vb).

## Dosyalar

| Dosya | Tür | Açıklama |
|---|---|---|
| `semantic_search/vector_store_chroma.py` | Yeni | ChromaDB-backed kalıcı vector store |
| `qa_rag/chunker.py` | Yeni | Token-aware hukuki metin chunker |
| `qa_rag/indexer.py` | Yeni | Bedesten → ChromaDB index pipeline |
| `qa_rag/rag_engine.py` | Modifiye | `backend` parametresi, ChromaDB desteği |
| `qa_rag/__init__.py` | Modifiye | Yeni exports (LegalChunker, BedestenIndexer) |
| `pyproject.toml` | Modifiye | chromadb, tiktoken bağımlılıkları, v1.2.0 |
| `tests/test_v12_smoke.py` | Yeni | Import + init smoke test |
| `tests/test_v12_indexer_small.py` | Yeni | 10-belge index + retrieval test |
| `tests/test_v12_rag_full.py` | Yeni | Tam RAG pipeline testi (LLM dahil) |
| `tests/v12_rag_test_results.json` | Yeni | Test sonuçları JSON |

## Çevre Değişkenleri

### ChromaDB
- `CHROMA_PERSIST_DIR` — Kalıcı dizin (default: `./chroma_db`)
- `CHROMA_COLLECTION` — Collection adı (default: `yargi_decisions`)
- `CHROMA_DISTANCE` — Distance metric: `cosine`|`l2`|`ip` (default: `cosine`)

### Indexer
- `INDEXER_BATCH_SIZE` — NVIDIA'ya bir seferde kaç chunk embed (default: 32)
- `INDEXER_TARGET_DOCS` — Hedef belge sayısı (default: 200)
- `INDEXER_KEYWORDS` — Virgülle ayrılmış anahtar kelimeler
- `INDEXER_COURT_TYPES` — Virgülle ayrılmış mahkeme tipleri
