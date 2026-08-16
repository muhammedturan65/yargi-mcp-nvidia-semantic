# Patch 006: Query Embedding Cache (LRU + ChromaDB Persistent) — v1.4.0

**Tarih:** 2026-08-17
**Sürüm:** v1.3.0 → v1.4.0
**Etkilenen dosyalar:** 6 (1 yeni, 5 modifiye)
**Toplam değişiklik:** +1243 / -69 satır

## Problem (v1.3.0'da kalan son darboğaz)

v1.3.0'da multi-provider LLM + answer cache ile tekrarlayan sorgular 1.86s'e
inmişti. Ama her sorguda hâlâ **~1 saniye NVIDIA query embedding** üretmek
zorundayız — answer cache HIT olduğunda bile!

Akış:
1. `LegalQARAG.retrieve(question)` →
2. `embedder.encode_query(question)` → **~1s NVIDIA API çağrısı** ← darboğaz
3. `ChromaDB.search_with_dedup(query_emb)` → ~50ms
4. `AnswerCache.lookup(question_emb)` → ~4ms (HIT ise LLM atlanır)

Yani:
- Answer cache HIT (LLM atlanır): retrieval yine ~1 s, çünkü NVIDIA'ya soru
  embedding'i için gidiliyor.
- Answer cache MISS: ~1 s (query embed) + 3-240s (LLM) — burada ~1 s
  önemsiz görünebilir ama her sorguda yineleniyor.

## Çözüm: İki Katmanlı Query Embedding Cache

```
Soru
  │
  ├─► normalize_query()  (ç→c, ı→i, ğ→g, lowercase, punct strip)
  │   "Mirasçı hangi davayı açar?" → "mirasci hangi davayi acar"
  │
  ├─► cache_key = sha256(normalized)[:16]  → "ecfa49a6aeaeb913"
  │
  ├─► LRU lookup (OrderedDict, 256 kayıt)  ~0 ms
  │   ├─ HIT → return cached embedding
  │   └─ MISS ↓
  │
  ├─► ChromaDB persistent lookup (by ID)   ~5 ms
  │   ├─ HIT → LRU'ya yaz, return
  │   └─ MISS ↓
  │
  └─► NVIDIA encode_query()  ~1 s
      + LRU'ya yaz + ChromaDB'ye upsert
```

### Neden exact-match cache, semantik değil?

NVIDIA `nv-embed-v1` (ve tüm OpenAI-compatible embedder'lar) **deterministik**
embedding üretir: aynı metin → aynı vektör. Yani "Mirasçı hangi davayı açar?"
her zaman aynı 4096-boyutlu vektöre mapping edilir.

Bu yüzden exact-match cache kullanmak doğru:
- **Query cache** → aynı soru → aynı embedding (exact match)
- **Answer cache** → semantik benzer soru → aynı cevap (cosine ≥ 0.92)

## Değiştirilen Dosyalar

### 1. YENİ: `qa_rag/query_cache.py` (426 satır)

İki katmanlı query embedding cache implementasyonu.

**Ana sınıflar:**
- `QueryEmbeddingCache`: ana cache sınıfı
  - `lookup(query)` → `Optional[QueryCacheHit]`
  - `store(query, embedding)` → `cache_key`
  - `clear()`, `size()`, `get_stats()`
- `QueryCacheHit`: dataclass (embedding, query, cache_key, source, cached_at)
- `normalize_query(query)`: Türkçe→ASCII + lowercase + punct strip
- `cache_key(query)`: SHA256(normalized)[:16]

**Cache key tasarımı:**
```python
# Türkçe karakterler → ASCII (önce, çünkü İ→i̇ combining dot)
_TR_ASCII_MAP = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})

def normalize_query(query: str) -> str:
    q = query.translate(_TR_ASCII_MAP)  # 1. Türkçe → ASCII
    q = q.lower()                        # 2. lowercase (artık güvenli)
    q = re.sub(r"[^\w\s]", " ", q)       # 3. punct → boşluk
    q = re.sub(r"\s+", " ", q).strip()   # 4. multi-space → single
    return q
```

**LRU implementasyonu:**
```python
self._lru: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()

def lookup(self, query):
    key = cache_key(query)
    if key in self._lru:
        # LRU: recently used → sona taşı
        self._lru.move_to_end(key)
        return self._lru[key]
    # ... persistent lookup, NVIDIA fallback
```

### 2. `qa_rag/rag_engine.py` (+91 / -18 satır)

**Değişiklikler:**
- `RAGContext`'e `query_cache_hit`, `query_cache_source` alanları
- `RAGResponse`'a `query_cache_hit`, `query_cache_source` alanları
- `LegalQARAG.__init__`'e `enable_query_cache` parametresi
- `_get_query_cache()` lazy-init metodu
- `retrieve()` metoduna cache lookup entegrasyonu:

```python
async def retrieve(self, question, top_k=None):
    query_cache = self._get_query_cache()
    cache_hit = query_cache.lookup(question)

    if cache_hit is not None:
        query_emb = cache_hit.embedding
        cache_source = cache_hit.source  # "lru" or "persistent"
    else:
        query_emb = embedder.encode_query(question)  # ~1s NVIDIA
        if query_cache.enabled:
            query_cache.store(question, query_emb)
        cache_source = "miss"

    # ... ChromaDB search
    return RAGContext(
        ...,
        query_cache_hit=(cache_source != "miss"),
        query_cache_source=cache_source,
    )
```

### 3. `qa_rag/__init__.py` (+11 satır)

Yeni exportlar:
```python
from .query_cache import QueryEmbeddingCache, QueryCacheHit, normalize_query, cache_key

__version__ = "1.4.0"  # 1.3.0'dan

__all__ = [
    ...,
    "QueryEmbeddingCache",
    "QueryCacheHit",
    "normalize_query",
    "cache_key",
    ...
]
```

### 4. `pyproject.toml` (+2 / -2 satır)

```diff
-version = "1.3.0"
-description = "... + Multi-provider LLM + Answer Cache"
+version = "1.4.0"
+description = "... + Multi-provider LLM + Answer Cache + Query Embedding Cache"

-keywords = [..., "answer-cache"]
+keywords = [..., "answer-cache", "query-cache", "lru-cache"]
```

### 5. `.env.example` (+17 satır)

```bash
# RAG QUERY EMBEDDING CACHE (v1.4.0+)
RAG_QUERY_CACHE=true
RAG_QUERY_CACHE_LRU_SIZE=256
RAG_QUERY_CACHE_COLLECTION=query_embed_cache
```

### 6. `CHANGELOG.md` (+72 satır)

v1.4.0 bölümü eklendi — tüm değişiklikler + benchmark + sınırlar.

## Before / After

### Retrieval Süresi (sadece NVIDIA query embed + ChromaDB search)

| Senaryo | v1.3.0 | v1.4.0 | İyileşme |
|---|---|---|---|
| İlk sorgu (cold) | ~1 s | ~1 s + cache yazma | ~ aynı |
| Aynı sorgu tekrar (LRU HIT) | ~1 s | **~5 ms** | **200x** |
| Aynı sorgu restart'tan sonra | ~1 s | **~11 ms** | **90x** |

### Tam RAG Pipeline (query embed + search + answer cache + LLM)

| Senaryo | v1.3.0 | v1.4.0 |
|---|---|---|
| Cold start (NVIDIA LLM) | 52.5 s | 67.3 s (aynı NATO) |
| Answer cache HIT (aynı process) | 1.86 s | **25 ms** |
| Answer cache HIT (yeni process) | 1.86 s | **11 ms** |
| **Toplam hızlanma** | **28.3x** (v1.2.0 → v1.3.0) | **~2700x** (cold → HIT) |

## Test Sonuçları

### 6/6 Smoke Test PASS

```
[1/6] Import testleri...                                  ✓
[2/6] normalize_query (Türkçe karakterler + punct)...     ✓
[3/6] QueryEmbeddingCache init (ChromaDB collection)...   ✓
[4/6] Store + LRU lookup + persistent lookup...           ✓
[5/6] Persistence (yeni instance, eski kayıtlar)...       ✓
[6/6] LegalQARAG retrieve() (gerçek NVIDIA + cache)...    ✓
```

### Cache HIT Benchmark (gerçek NVIDIA LLM)

```
Cold start (NVIDIA query + NVIDIA LLM): 67.3 s   (Q=MISS, A=MISS)
Aynı process (LRU + answer cache HIT):    25 ms   (Q=persistent, A=HIT)
Yeni process (persistent + A-cache HIT):  11 ms   (Q=persistent, A=HIT)
```

Hızlanma: cold → HIT = **~2700x**

### Test dosyaları

- `tests/v14_query_cache_results.json` — 6/6 smoke test
- `tests/v14_cache_hit_results.json` — cache HIT benchmark
- `tests/v14_full_rag_results.json` — full RAG pipeline (cold start)

## Çevre Değişkenleri Referansı

| Env Var | Default | Açıklama |
|---|---|---|
| `RAG_QUERY_CACHE` | `true` | Query cache açık/kapalı |
| `RAG_QUERY_CACHE_LRU_SIZE` | `256` | In-memory LRU max kayıt |
| `RAG_QUERY_CACHE_COLLECTION` | `query_embed_cache` | ChromaDB collection adı |

## Bilinen Sınırlar (v1.4.0)

- **Exact-match only**: Cache key normalize edilmiş exact match'tir (semantik değil).
  "Mirasçı hangi davayı açar?" ve "Mirasçılar hangi davayı açar?" farklı cache kayıtları
  olur. Bu kasıtlı: semantik benzerlik `AnswerCache`'in işi.

- **LRU default 256**: Düşük QPS senaryosu için yeterli. Yüksek QPS için
  `RAG_QUERY_CACHE_LRU_SIZE` ile artırılabilir.

- **TTL yok**: Hukuki soruların embedding'i değişmez (NVIDIA model sabit),
  bu yüzden TTL gerekmez. Model değişirse `cache.clear()` ile temizlenmeli.

## Sonraki Adımlar (v1.5.0 adayları)

- Next.js demo dashboard (RAG arayüzü)
- yargi-cli TypeScript port'a semantik arama + RAG desteği
- Section-aware retrieval scoring (GEREKÇE section'ına ağırlık)
- Çok-dilli destek (Türkçe + İngilizce + Almanca hukuki metinler)
