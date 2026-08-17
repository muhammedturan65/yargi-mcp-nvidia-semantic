"""Quick smoke test: tüm yeni modüller import edilebiliyor mu?"""
import sys, os
sys.path.insert(0, "/home/z/my-project/repos/yargi-mcp-nvidia-semantic")

# NVIDIA env
NVIDIA_KEY = "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"
os.environ["NVIDIA_API_KEY"] = NVIDIA_KEY
os.environ["LOCAL_EMBEDDING_API_KEY"] = NVIDIA_KEY
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LOCAL_EMBEDDING_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["LOCAL_EMBEDDING_MODEL"] = "nvidia/nv-embed-v1"
os.environ["LOCAL_EMBEDDING_DIMENSION"] = "4096"
os.environ["LOCAL_EMBEDDING_INPUT_TYPE"] = "auto"
os.environ["EMBEDDING_PROMPT_STYLE"] = "raw"
os.environ["CHROMA_PERSIST_DIR"] = "/home/z/my-project/repos/yargi-mcp-nvidia-semantic/chroma_db"
os.environ["CHROMA_COLLECTION"] = "yargi_decisions"

print("1. Import test: qa_rag module...")
from qa_rag import (
    LegalQARAG, RAGResponse, RAGContext,
    NvidiaLLMClient, SYSTEM_PROMPT_LEGAL, build_user_prompt,
    LegalChunker, Chunk, chunk_text,
    BedestenIndexer, IndexResult, IndexProgress,
)
print("   OK - qa_rag tüm import'lar başarılı")
print(f"   qa_rag.__version__ = {__import__('qa_rag').__version__}")

print("\n2. Import test: ChromaVectorStore...")
from semantic_search.vector_store_chroma import ChromaVectorStore, _ChromaDocument, _sanitize_chroma_meta
print("   OK - ChromaVectorStore import edildi")

print("\n3. ChromaVectorStore init test (boş)...")
store = ChromaVectorStore(dimension=4096, collection_name="yargi_decisions")
print(f"   OK - Collection: {store.collection_name}")
print(f"   Mevcut kayıt sayısı: {store.size()}")
stats = store.get_stats()
print(f"   Stats: {stats}")

print("\n4. LegalQARAG init test (chroma backend)...")
rag = LegalQARAG(backend="chroma")
print(f"   OK - backend: {rag.backend}")
print(f"   is_corpora_loaded (henüz init yok): {rag.is_corpora_loaded}")

print("\n5. Embedder init test...")
emb = rag._get_embedder()
print(f"   OK - Embedder: {emb.model}, {emb.dimension}d")

print("\n6. VectorStore auto-init test...")
vs = rag._get_vector_store()
print(f"   OK - VectorStore: {type(vs).__name__}, size={vs.size()}")

print("\n7. Chunker init test...")
from qa_rag.chunker import LegalChunker
chunker = LegalChunker()
print(f"   OK - target_tokens={chunker.target_tokens}, overlap={chunker.overlap_tokens}")

print("\n8. Tokenizer check...")
from qa_rag.chunker import count_tokens, _TOKENIZER_NAME
print(f"   Tokenizer: {_TOKENIZER_NAME}")
print(f"   count_tokens('Merhaba dünya, bu bir test cümlesidir.') = {count_tokens('Merhaba dünya, bu bir test cümlesidir.')}")

print("\n9. ChromaDB metadata sanitize test...")
meta = {
    "document_id": "ABC123",
    "esas_no": "E.2026/2403",
    "list_field": [1, 2, 3],
    "none_field": None,
    "bool_field": True,
    "dict_field": {"nested": "value"},
}
clean = _sanitize_chroma_meta(meta)
print(f"   Input:  {meta}")
print(f"   Output: {clean}")
assert "none_field" not in clean, "None field atlanmalı"
assert isinstance(clean["list_field"], str), "List field string'e çevrilmeli"
assert isinstance(clean["dict_field"], str), "Dict field string'e çevrilmeli"
print("   OK - sanitize çalışıyor")

print("\n=== TÜM SMOKE TESTLERİ GEÇTİ ===")
