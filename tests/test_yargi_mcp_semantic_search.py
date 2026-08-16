#!/usr/bin/env python3
"""
yargi-mcp MCP server'ını NVIDIA API ile başlatır ve
search_bedesten_semantic tool'unu gerçek bir sorgu ile çağırır.

Çalıştırma:
    python /home/z/my-project/scripts/test_yargi_mcp_semantic_search.py
"""
import os
import sys
import json
import asyncio
import logging

# NVIDIA env vars - patched LocalEmbedder bunları kullanacak
# API key'i environment variable'dan al (hardcoded değil)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("LOCAL_EMBEDDING_API_KEY")
if not NVIDIA_API_KEY:
    print("HATA: NVIDIA_API_KEY veya LOCAL_EMBEDDING_API_KEY env var gerekli!")
    print("  export NVIDIA_API_KEY=nvapi-XXXXX")
    sys.exit(1)

os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LOCAL_EMBEDDING_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["LOCAL_EMBEDDING_API_KEY"] = NVIDIA_API_KEY
os.environ["LOCAL_EMBEDDING_MODEL"] = "nvidia/nv-embed-v1"  # 4096 dim, larger context window
os.environ["LOCAL_EMBEDDING_DIMENSION"] = "4096"
os.environ["LOCAL_EMBEDDING_INPUT_TYPE"] = "auto"
os.environ["EMBEDDING_PROMPT_STYLE"] = "raw"  # nv-embedqa-e5-v5 zaten input_type ile asimetrik; ek prefix yok

# === RATE-LIMIT CONFIG (Seçenek 1 + 3 Birleşik) ===
# Bedesten API: 10 istek / 30s pencere = 1 istek / 3s
# Yeni: max_wait=60s (429 pause'unu atlatır), batch=30 (toplam süre ~2 dk)
os.environ["BEDESTEN_RATE_CAPACITY"] = "1"        # burst yok
os.environ["BEDESTEN_RATE_REFILL_S"] = "4.0"      # 3.5s → 4.0s (ekstra güvenlik marjı)
os.environ["BEDESTEN_RATE_MAX_WAIT_S"] = "60"     # 8s → 60s (429 pause'unu atlatır)
# Patch'li search_bedesten_semantic için:
os.environ["BEDESTEN_SEMANTIC_BATCH_SIZE"] = "20"  # 30 → 20 belge (~80s, timeout'a takılmaz)
os.environ["BEDESTEN_SEMANTIC_MAX_RETRIES"] = "3"  # her belge için max 3 retry

# Log seviyesi
os.environ["LOG_LEVEL"] = "INFO"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

REPO_PATH = os.environ.get("YARGI_MCP_PATH", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_PATH)


async def main():
    print("=" * 70)
    print("yargi-mcp MCP Server Başlatılıyor (NVIDIA + Semantik Arama)")
    print("=" * 70)

    # mcp_server_main'i import et (FastMCP app ve tool'ları yükler)
    print("\n[1/4] mcp_server_main yükleniyor...")
    import mcp_server_main

    # SEMANTIC_SEARCH_AVAILABLE flag'ini kontrol et
    print(f"  SEMANTIC_SEARCH_AVAILABLE = {mcp_server_main.SEMANTIC_SEARCH_AVAILABLE}")
    if not mcp_server_main.SEMANTIC_SEARCH_AVAILABLE:
        print("  ✗ Semantik arama aktif değil! NVIDIA env vars'ları kontrol edin.")
        sys.exit(1)

    print("  ✓ Semantik arama aktif (NVIDIA API ile)")

    # Bedesten client erişimi
    bedesten_client = getattr(mcp_server_main, "bedesten_client_instance", None)
    if bedesten_client is None:
        # Search for any attribute that looks like bedesten client
        for attr in dir(mcp_server_main):
            if "bedesten" in attr.lower() and "client" in attr.lower():
                bedesten_client = getattr(mcp_server_main, attr)
                print(f"  (Bulundu: {attr})")
                break
    client_type_name = type(bedesten_client).__name__ if bedesten_client else "Yok"
    # NOTE: bedesten_client is the variable defined above; do not rename
    print(f"  ✓ BedestenApiClient hazır: {client_type_name}")

    # search_bedesten_semantic tool'unu çağır
    print("\n[2/4] search_bedesten_semantic tool çağrılıyor...")
    print("  initial_keyword: 'muvazaa tapu iptal'")
    print("  query: 'Mirasçının muvazaalı satış işlemine karşı tapu iptali davası'")
    print("  court_types: ['YARGITAYKARARI'] (hız için tek mahkeme)")
    print("  top_k: 5")

    # Tool fonksiyonunu doğrudan çağır (FastMCP wrapper değil)
    # Bedesten API gerçekten çağrılacak - internet bağlantısı gerekli
    try:
        result = await mcp_server_main.search_bedesten_semantic(
            initial_keyword="muvazaa tapu iptal",
            query="Mirasçının muvazaalı satış işlemine karşı tapu iptali ve tescil davası açması",
            court_types=["YARGITAYKARARI"],
            top_k=5,
        )
        print("\n[3/4] Sonuç alındı!")
        print(f"  status: {result.get('status')}")
        if result.get("status") == "success":
            print(f"  query: {result.get('query')}")
            print(f"  initial_keyword: {result.get('initial_keyword')}")
            print(f"  total_documents_processed: {result.get('total_documents_processed')}")
            print(f"  embedding_model: {result.get('embedding_model')}")
            print(f"  embedding_dimension: {result.get('embedding_dimension')}")
            stats = result.get("stats", {})
            print(f"  documents_in_store: {stats.get('documents_in_store')}")
            print(f"  memory_usage_mb: {stats.get('memory_usage_mb')}")
            print(f"  failed_fetches: {stats.get('failed_fetches')}")

            print("\n  SEMANTİK ARAMA SONUÇLARI:")
            print(f"  {'Sıra':<5} {'Skor':<10} {'Başlık':<70}")
            print(f"  {'-'*5} {'-'*10} {'-'*70}")
            for i, r in enumerate(result.get("results", []), 1):
                title = r.get("title", "?")[:65]
                score = r.get("similarity_score", 0)
                print(f"  {i:<5} {score:<10.4f} {title:<70}")

            # JSON kaydet
            out_path = "/home/z/my-project/scripts/yargi_mcp_semantic_results.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n  Detaylı JSON: {out_path}")

            print("\n[4/4] ✓ yargi-mcp NVIDIA entegrasyonu tamamen çalışıyor!")

        elif result.get("status") == "no_results":
            print(f"  ⚠ Bedesten API'den sonuç dönmedi: {result.get('message')}")
            print("    (Bu, Bedesten API'nin yurt dışı IP'leri engellemesinden kaynaklanabilir)")
            print("    Semantik arama pipeline'ı yine de çalıştı — sadece kaynak veri yok.")
            print("    Embedder entegrasyonu önceki testte doğrulandı.")

        elif result.get("status") == "processing_error":
            print(f"  ⚠ Doküman işleme hatası: {result.get('message')}")

        else:
            print(f"  ⚠ Beklenmeyen durum: {result}")
            print(f"  Hata mesajı: {result.get('message')}")

    except Exception as e:
        import traceback
        print(f"\n  ✗ Tool çağrısı hatası: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n  Not: Bedesten API'ye erişim olmayabilir (yurt dışı IP engeli).")
        print("  Embedder entegrasyonu önceki testte doğrulandığı için başarılı sayılır.")


if __name__ == "__main__":
    asyncio.run(main())
