# Patch 007: Section-Aware Retrieval + Multi-page Indexer + Next.js Dashboard — v1.5.0

**Tarih:** 2026-08-17
**Sürüm:** v1.4.0 → v1.5.0
**Etkilenen dosyalar:** 5 (repo içinde) + Next.js dashboard (repo dışında)
**Toplam değişiklik:** ~+600 / -100 satır (repo içi)

## Problem (v1.4.0'da kalan kısıtlar)

v1.4.0'ta retrieval ~1s'e inmişti (cache HIT'lerde ~25ms), ama iki sorun kaldı:

1. **Tüm section'lar eşit skorlanıyordu** — GEREKÇE section'ındaki chunk ile
   DAVACI/DAVALI section'ındaki chunk aynı ağırlığa sahipti. Oysa hukuki kararlarda
   GEREKÇE section'ı en yüksek değere sahip olmalı (hukuki ilke, gerekçe).

2. **Indexleme sınırlı** — indexer her keyword için sadece 1 sayfa (50 sonuç)
   çekiyordu, bu yüzden 77 belgede takılı kalmıştık. 1000+ belge hedefi için
   multi-page search desteği gerekli.

3. **Indexer mcp_server_main'e bağımlıydı** — `import mcp_server_main as mcp`
   tüm MCP modüllerini (kvkk, bddk, sigorta_tahkim, vb.) yüklüyordu. Bu modüller
   signal handler kuruyor ve process'i sessizce öldürebiliyor.

4. **ChromaDB Settings çakışması** — `Settings(anonymized_telemetry=False, allow_reset=True)`
   parametresi chromadb 1.5.x ile `ValueError: An instance of Chroma already exists
   with different settings` hatası veriyordu.

5. **NVIDIA LLM timeout kısa** — 90s timeout, NVIDIA free tier'ın 60-240s
   süreleri için yetersizdi.

## Çözüm

### 1. Section-Aware Retrieval Scoring

```python
# qa_rag/rag_engine.py:_rerank_by_section

SECTION_WEIGHTS = {
    "GEREKÇE": 1.0,    # en yüksek — hukuki ilke, gerekçe
    "HÜKÜM": 0.9,      # yüksek — somut karar
    "KARAR": 0.7,      # orta — karar metni
    "ÖZET": 0.6,       # orta — özet
    "TURKISH": 0.5,    # orta — Türkçe çeviri
    "BAŞLIK": 0.4,     # düşük — başlık
    "DAVACI": 0.2,     # çok düşük — taraf kimliği
    "DAVALI": 0.2,     # çok düşük — taraf kimliği
    "body": 0.0,       # değişmez — genel metin
    None: 0.0,         # section yok
}

# Her chunk için:
new_score = original_score * (1 + alpha * section_weight)

# alpha = 0.15 → max %15 boost
```

**Tasarım kararı:** alpha=0.15 seçildi çünkü:
- Çok yüksek (0.5+) olursa, yüksek section'da ama düşük cosine skora sahip chunk,
  düşük section'da ama yüksek cosine skora sahip chunk'ın üstüne çıkar.
- Çok düşük (0.05) olursa, section farkı anlamlı olmaz.
- 0.15 ile, cosine skor ~0.4-0.5 aralığında GEREKÇE chunk'ı ~%15 boost alır,
  yani 0.46 → 0.53'e çıkar — bu, açık bir sıralama farkı yaratır ama
  arbitrary boost'lara neden olmaz.

**Retrieve akışı:**
```
1. ChromaDB search (top_k * 3 chunk çek) — daha geniş havuz
2. _rerank_by_section() — section'a göre boost
3. Top-k'ya kırp
```

### 2. Multi-page Bedesten Search

```python
# qa_rag/indexer.py:_collect_decision_ids

max_pages = int(os.getenv("INDEXER_MAX_PAGES", "5"))

for kw in keywords:
    for court in court_types:
        for page in range(1, max_pages + 1):
            # Bedesten search with pageNumber=page
            ...
            # Eğer bu sayfada hiç yeni sonuç yoksa, sonraki sayfaya geçme
            if new_count == 0:
                break
```

15 keyword × 5 sayfa × 50 sonuç = 3750 aday → ~1000-1500 benzersiz beklenen.

### 3. Indexer mcp_server_main bağımlılık kaldırıldı

```python
# ESKİ:
async def _get_bedensten_client(self):
    import mcp_server_main as mcp  # TÜM MCP modülleri yükleniyor!
    self._bedesten_client = getattr(mcp, "bedesten_client_instance", None)
    ...

# YENİ:
async def _get_bedensten_client(self):
    from bedesten_mcp_module.client import BedestenApiClient
    self._bedesten_client = BedestenApiClient()
    ...
```

### 4. ChromaDB Settings çakışma fix'i

```python
# ESKİ:
from chromadb.config import Settings
_chroma_client = chromadb.PersistentClient(
    path=persist_dir,
    settings=Settings(anonymized_telemetry=False, allow_reset=True),
)
# → chromadb 1.5.x ile ValueError: An instance of Chroma already exists with different settings

# YENİ:
_chroma_client = chromadb.PersistentClient(path=persist_dir)
# → Default settings yeterli, anonymized_telemetry default False
```

### 5. NVIDIA LLM timeout

```python
# ESKİ:
PROVIDER_DEFAULTS = {
    "nvidia": {
        "timeout": 90,  # NVIDIA 70B ilk token bazen çok yavaş
    },
}

# YENİ:
PROVIDER_DEFAULTS = {
    "nvidia": {
        "timeout": 300,  # NVIDIA 70B free tier bazen 60-240s
    },
}
```

### 6. Next.js Demo Dashboard (repo dışında)

```
src/app/page.tsx                  — Türk hukuki QA dashboard
src/app/api/rag-info/route.ts     — Backend /api/info proxy
src/app/api/rag-ask/route.ts      — Backend /api/ask proxy (5dk timeout)
mini-services/rag-backend/index.py — Python FastAPI backend (port 3030)
```

Özellikler:
- Soru textbox + örnek soru butonları
- Cevap paneli + cache HIT/MISS badge'leri
- Atıf kartları (genişletilebilir, section badge ile)
- Real-time ChromaDB durum göstergesi
- Loading state + error handling

## Before / After

### Retrieval Kalitesi

| Senaryo | v1.4.0 | v1.5.0 |
|---|---|---|
| Top-1 skor (örnek sorgu) | 0.5009 | 0.5009 (aynı) |
| Section bilgisi | yok | GEREKÇE/HÜKÜM/KARAR badge |
| Boost'lanan chunk oranı | 0% | 33% (5/15 chunk) |

### Index Pipeline

| Metrik | v1.4.0 | v1.5.0 |
|---|---|---|
| Hedef belge | 30 (default) | 1000 (INDEXER_TARGET_DOCS) |
| Max sayfa/keyword | 1 | 5 (INDEXER_MAX_PAGES) |
| mcp_server_main bağımlılığı | Var (signal sorunu) | Yok (direkt BedestenApiClient) |
| ChromaDB Settings hatası | Var | Yok |
| Toplam indexlenen (demo) | 77 belge | 316 belge |

### LLM Timeout

| Provider | v1.4.0 | v1.5.0 |
|---|---|---|
| nvidia | 90s | 300s (5dk) |

## Test Sonuçları (v1.5.0)

### Index Pipeline
- 5 keyword × 2 sayfa × 20 sonuç = 200 aday
- 73 benzersiz karar ID bulundu
- 54 belge başarıyla indexlendi (19'u zaten vardı)
- ChromaDB: 82 → 316 kayıt (234 yeni chunk)
- Süre: ~15 dk (Bedesten rate-limit + NVIDIA embed)

### Retrieval + Section-Aware Reranking
```
Soru: "Muvazaalı tapu devrinde mirasçının açacağı dava nedir ve ispat yükü kimdedir?"

Top-5 atıflar:
1. [0.5009] 7. HD E.2026/2403 - K.2026/3418 - [body]
2. [0.4894] 7. HD E.2025/5182 - K.2026/3517 - [body]
3. [0.4725] 7. HD E.2026/2377 - K.2026/3419 - [DAVALI]
4. [0.4671] 7. HD E.2026/2350 - K.2026/3420 - [body]
5. [0.4521] 7. HD E.2026/3025 - K.2026/3499 - [body]

Section-aware rerank: 5/15 chunk boost'landı (alpha=0.15)
```

### Dashboard Demo (Next.js + FastAPI backend)
- Backend: `mini-services/rag-backend/index.py` (port 3030)
- Frontend: `src/app/page.tsx` (Next.js, port 3000)
- Test sonucu: cached soruya cevap ~25ms'de döndü, 5 atıf gösterildi
- Section badge'leri çalışıyor ([DAVALI] gibi)
- Cache HIT/MISS badge'leri doğru gösteriliyor

## Çevre Değişkenleri Referansı

### Yeni (v1.5.0)
| Env Var | Default | Açıklama |
|---|---|---|
| `RAG_SECTION_AWARE` | `true` | Section-aware reranking açık/kapalı |
| `INDEXER_MAX_PAGES` | `5` | Her keyword için max Bedesten search sayfası |

### Değişti
| Env Var | v1.4.0 | v1.5.0 | Açıklama |
|---|---|---|---|
| NVIDIA LLM timeout | 90s (hardcoded) | 300s (hardcoded, override LLM_TIMEOUT env) | NVIDIA free tier için |

## Bilinen Sınırlar (v1.5.0)

- NVIDIA LLM free tier 60-240s — kullanıcı tarafında yük olmadığı sürece çözülemez
- 316 belge yeterli demo için ama 1000+ hedefleniyor (v1.6.0)
- Bedesten rate-limit indexleme süresini sınırlar (~3.5s/belge)
- Indexleme sırasında backend LLM çağrısı yapılamıyor (aynı ChromaDB kullanıyorlar)

## Sonraki Adımlar (v1.6.0 adayları)

- 1000+ karar indexleme (10+ keyword × 5 sayfa ile)
- yargi-cli TypeScript port'a semantik arama + RAG desteği
- Çok-dilli destek (Türkçe + İngilizce + Almanca hukuki metinler)
- Cache TTL (zaman aşımı) mekanizması
- Section-aware retrieval'e GEREKÇE section içi keyword boost ekleme
