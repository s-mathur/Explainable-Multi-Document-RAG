# =====================================================
# DOCUMENT CONFIGURATION
# =====================================================

# Maximum allowed document size in MB.
MAX_FILE_SIZE_MB = 10

# Supported file formats.
SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx"]


# =====================================================
# DOCUMENT EXPIRY CONFIGURATION
# =====================================================

# Time-to-live for uploaded documents.
# Default = 30 minutes.
DOCUMENT_TTL_SECONDS = 30 * 60


# =====================================================
# CHUNKING CONFIGURATION
# =====================================================

# Number of characters per chunk.
CHUNK_SIZE = 500

# Number of overlapping characters.
CHUNK_OVERLAP = 100


# =====================================================
# RETRIEVAL CONFIGURATION
# =====================================================

# Number of chunks to retrieve from FAISS.
TOP_K = 3


# =====================================================
# EMBEDDING CONFIGURATION
# =====================================================

# Sentence Transformer model.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Embedding dimensions for the model above.
EMBEDDING_DIMENSION = 384


# =====================================================
# VECTOR DATABASE CONFIGURATION
# =====================================================

# Used only for displaying in the UI.
VECTOR_DB = "FAISS"


# =====================================================
# LLM CONFIGURATION
# =====================================================

# Gemini model used for response generation.
LLM_MODEL = "gemini-2.5-flash"


# =====================================================
# STORAGE CONFIGURATION
# =====================================================

# Root directory for storing user sessions.
USER_DOCUMENTS_DIR = "user_documents"

# Metadata file name.
METADATA_FILE = "metadata.json"

# FAISS index file name.
FAISS_INDEX_FILE = "index.faiss"

# Chunk metadata file.
CHUNK_METADATA_FILE = "chunks.pkl"

# Embedding metadata file.
EMBEDDING_METADATA_FILE = "embeddings.pkl"


# =====================================================
# UI CONFIGURATION
# =====================================================

# Show vectors only when requested by the user.
SHOW_EMBEDDINGS_BY_DEFAULT = False

# Show chunk text only when requested by the user.
SHOW_CHUNK_TEXT_BY_DEFAULT = False

# Query embedding visibility.
SHOW_QUERY_EMBEDDING_BY_DEFAULT = False


# =====================================================
# RAG CONFIGURATION SUMMARY
# =====================================================

RAG_CONFIGURATION = {
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dimension": EMBEDDING_DIMENSION,
    "vector_database": VECTOR_DB,
    "llm_model": LLM_MODEL,
    "document_ttl_seconds": DOCUMENT_TTL_SECONDS
}
