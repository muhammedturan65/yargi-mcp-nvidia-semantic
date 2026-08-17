#!/usr/bin/env python3
"""
yargi-qa — Türk Hukuki QA Chatbot CLI

RAG pipeline: Bedesten emsal kararları + NVIDIA nv-embed-v1 + NVIDIA Nemotron LLM

Kullanım:
    # 1) NVIDIA API key set et
    export NVIDIA_API_KEY=nvapi-...

    # 2) Interaktif REPL başlat
    python -m qa_rag.cli
    # veya
    yargi-qa

    # 3) Tek soru modu
    python -m qa_rag.cli --ask "Mirasçı muvazaalı satışa karşı hangi davayı açar?"

    # 4) Streaming modu (token token yaz)
    python -m qa_rag.cli --ask "..." --stream

İlk çalıştırmada Bedesten API'den 30 karar çeker (~2 dk, rate-limit).
Sonraki sorular hızlı çalışır (saniyeler içinde).

CLI komutları (REPL modunda):
    /load <keyword>   Yeni bir konu için corpus yükle (örn: /load nafaka)
    /info             Mevcut corpus bilgisi
    /clear            Ekranı temizle
    /quit             Çık
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import textwrap

# Repo root path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Hukuki soru örnekleri (yeni kullanıcı için ipucu)
EXAMPLE_QUESTIONS = [
    "Mirasçı, murisin muvazaalı satış işlemine karşı hangi davayı açabilir?",
    "Tapu iptal ve tescil davasında dava açma süresi nedir?",
    "Muvazaa iddiasında ispat yükü kimdedir?",
    "Gizli kazanç iddiası hangi koşullarda geçerlidir?",
    "Mirasçılık sıfatı ne zaman kazanılır?",
]


def print_banner():
    print("=" * 72)
    print("  yargi-qa — Türk Hukuki QA Chatbot (RAG)")
    print("  NVIDIA nv-embed-v1 + Nemotron LLM + Bedesten emsal kararları")
    print("=" * 72)
    print()


def print_help():
    print("""
Komutlar:
  /load <keyword>   Yeni corpus yükle (örn: /load nafaka)
  /info             Mevcut corpus bilgisi
  /examples         Örnek hukuki soruları göster
  /clear            Ekranı temizle
  /help             Bu yardım
  /quit             Çıkış

Normal bir hukuki soru yazıp Enter'a basın. Cevap emsal karar referanslarıyla
birlikte üretilecek.
""")


def print_info(rag):
    vs = rag._vector_store
    embedder = rag._embedder
    llm = rag._llm_client

    print()
    print("─" * 60)
    print("  Corpus Bilgisi")
    print("─" * 60)
    if vs:
        stats = vs.get_stats()
        print(f"  Vector store boyutu:    {stats['num_documents']} karar")
        print(f"  Embedding boyutu:       {stats['dimension']}d")
        print(f"  Memory kullanımı:       {stats['memory_usage_mb']:.2f} MB")
    else:
        print("  Vector store:           (boş)")
    if embedder:
        print(f"  Embedding modeli:       {embedder.model}")
    if llm:
        print(f"  LLM modeli:             {llm.model}")
        print(f"  LLM temperature:        {llm.temperature}")
    print("─" * 60)
    print()


def print_examples():
    print("\nÖrnek hukuki sorular:\n")
    for i, q in enumerate(EXAMPLE_QUESTIONS, 1):
        print(f"  {i}. {q}")
    print()


def format_response(response) -> str:
    """RAGResponse'u güzel bir şekilde formatla."""
    lines = []
    lines.append("─" * 72)
    lines.append(f"Soru: {response.question}")
    lines.append("─" * 72)
    lines.append("")
    lines.append(response.answer)
    lines.append("")
    lines.append("─" * 72)
    lines.append("Kaynak Kararlar:")
    for i, c in enumerate(response.citations, 1):
        md = c.get("metadata", {})
        parts = [f"[{i}]"]
        if md.get("birim_adi"):
            parts.append(md["birim_adi"])
        if md.get("esas_no"):
            parts.append(f"E.{md['esas_no']}")
        if md.get("karar_no"):
            parts.append(f"K.{md['karar_no']}")
        if md.get("karar_tarihi"):
            parts.append(f"({md['karar_tarihi']})")
        score = c.get("similarity_score", 0)
        parts.append(f"[skor: {score:.4f}]")
        lines.append("  " + " ".join(parts[1:]))

    lines.append("")
    lines.append("─" * 72)
    lines.append(
        f"⏱  Toplam: {response.total_time_ms:.0f}ms "
        f"(retrieval: {response.retrieval_time_ms:.0f}ms, "
        f"LLM: {response.generation_time_ms:.0f}ms) | "
        f"Tokens: {response.llm_usage.get('total_tokens', 0)} "
        f"(prompt: {response.llm_usage.get('prompt_tokens', 0)}, "
        f"completion: {response.llm_usage.get('completion_tokens', 0)})"
    )
    lines.append("─" * 72)
    return "\n".join(lines)


async def stream_response(rag, question: str):
    """Streaming RAG — token token yaz."""
    print("─" * 72)
    print(f"Soru: {question}")
    print("─" * 72)
    print()
    print("Cevap (streaming): ", end="", flush=True)

    async for chunk in rag.ask_stream(question):
        print(chunk, end="", flush=True)

    print()
    print()
    print("─" * 72)
    print("(Streaming tamamlandı. Atıflar için non-stream modu kullanın.)")
    print("─" * 72)


async def repl_loop(rag, args):
    """Ana interaktif REPL."""
    print_banner()

    # İlk corpus load (eğer --skip-load yoksa)
    if not rag.is_corpora_loaded and not args.skip_load:
        print("İlk kurulum: Bedesten'den 30 karar yükleniyor (~2 dk)...")
        print("  keyword: 'muvazaa tapu iptal'")
        print()
        try:
            result = await rag.load_corpora()
            stats = result.get("stats", {})
            print(f"\n✓ {stats.get('documents_in_store', 0)} karar yüklendi "
                  f"({stats.get('failed_fetches', 0)} fetch hatası)")
            print()
        except Exception as e:
            print(f"\n✗ Corpus yükleme hatası: {e}")
            print("  /load <keyword> ile farklı bir konu deneyin.")
            print()

    print_help()

    while True:
        try:
            user_input = input("yargi-qa> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkış...")
            break

        if not user_input:
            continue

        # Komutlar
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("quit", "exit", "q"):
                print("Çıkış...")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "info":
                print_info(rag)
            elif cmd == "examples":
                print_examples()
            elif cmd == "clear":
                os.system("clear" if os.name == "posix" else "cls")
            elif cmd == "load":
                if not arg:
                    print("Kullanım: /load <keyword>")
                    continue
                print(f"Yeni corpus yükleniyor: '{arg}' (~2 dk)...")
                try:
                    result = await rag.load_corpora(
                        initial_keyword=arg,
                        semantic_query=arg,  # basit
                    )
                    stats = result.get("stats", {})
                    print(f"✓ {stats.get('documents_in_store', 0)} karar yüklendi")
                except Exception as e:
                    print(f"✗ Hata: {e}")
            else:
                print(f"Bilinmeyen komut: /{cmd}. /help yazın.")
            continue

        # Normal soru
        try:
            if args.stream:
                await stream_response(rag, user_input)
            else:
                print("Düşünüyor...")
                response = await rag.ask(user_input)
                print()
                print(format_response(response))
                print()
        except Exception as e:
            print(f"\n✗ Hata: {type(e).__name__}: {e}")
            import traceback
            if args.debug:
                traceback.print_exc()
            print()


async def main_async():
    parser = argparse.ArgumentParser(
        description="yargi-qa — Türk Hukuki QA Chatbot (RAG pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Örnekler:
            yargi-qa                                  # interaktif REPL
            yargi-qa --ask "Muvazaalı satışa karşı hangi dava?"
            yargi-qa --ask "..." --stream             # streaming cevap
            yargi-qa --skip-load                      # corpus yüklemeden başla
        """),
    )
    parser.add_argument("--ask", type=str, help="Tek soru (REPL yerine)")
    parser.add_argument("--stream", action="store_true", help="Streaming modu")
    parser.add_argument("--skip-load", action="store_true",
                        help="Başlangıçta corpus yükleme (varsa)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="LLM'e kaç karar feed'lenecek (default: 5)")
    parser.add_argument("--n-decisions", type=int, default=30,
                        help="Corpus için kaç karar çekilecek (default: 30)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature (default: 0.2)")
    parser.add_argument("--debug", action="store_true", help="Traceback göster")
    args = parser.parse_args()

    # Logging — INFO seviyesinde konsola yaz
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # NVIDIA API key kontrolü
    if not (os.environ.get("NVIDIA_API_KEY") or os.environ.get("LOCAL_EMBEDDING_API_KEY")):
        print("HATA: NVIDIA_API_KEY env var gerekli!")
        print("  export NVIDIA_API_KEY=nvapi-XXXXX")
        sys.exit(1)

    # Bedesten rate-limit config (load_corpora için)
    os.environ.setdefault("BEDESTEN_RATE_CAPACITY", "1")
    os.environ.setdefault("BEDESTEN_RATE_REFILL_S", "4.0")
    os.environ.setdefault("BEDESTEN_RATE_MAX_WAIT_S", "60")
    os.environ.setdefault("BEDESTEN_SEMANTIC_BATCH_SIZE", str(args.n_decisions))
    os.environ.setdefault("BEDESTEN_SEMANTIC_MAX_RETRIES", "3")

    # Embedding config (yoksa default)
    os.environ.setdefault("EMBEDDING_PROVIDER", "local")
    os.environ.setdefault("LOCAL_EMBEDDING_BASE_URL", "https://integrate.api.nvidia.com/v1")
    os.environ.setdefault("LOCAL_EMBEDDING_MODEL", "nvidia/nv-embed-v1")
    os.environ.setdefault("LOCAL_EMBEDDING_DIMENSION", "4096")
    os.environ.setdefault("LOCAL_EMBEDDING_INPUT_TYPE", "auto")
    os.environ.setdefault("EMBEDDING_PROMPT_STYLE", "raw")

    # RAG engine
    from qa_rag.rag_engine import LegalQARAG
    rag = LegalQARAG(
        n_decisions_per_query=args.n_decisions,
        top_k_retrieval=args.top_k,
        llm_temperature=args.temperature,
    )

    # Tek soru modu
    if args.ask:
        if not rag.is_corpora_loaded:
            print("Corpus yükleniyor (~2 dk)...")
            await rag.load_corpora()

        if args.stream:
            await stream_response(rag, args.ask)
        else:
            response = await rag.ask(args.ask)
            print(format_response(response))
        return

    # REPL
    await repl_loop(rag, args)


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nÇıkış...")
        sys.exit(0)


if __name__ == "__main__":
    main()
