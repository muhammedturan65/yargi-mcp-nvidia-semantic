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
