import json
import os
import pickle
import shutil
import time
import uuid
from datetime import datetime, timedelta

from config import (
    USER_DOCUMENTS_DIR,
    METADATA_FILE,
    CHUNK_METADATA_FILE,
    EMBEDDING_METADATA_FILE,
    DOCUMENT_TTL_SECONDS
)


# =====================================================
# SESSION FUNCTIONS
# =====================================================

def create_user_session():
    """
    Creates a unique user session directory.
    """
    session_id = str(uuid.uuid4())
    session_path = os.path.join(USER_DOCUMENTS_DIR, session_id)

    os.makedirs(session_path, exist_ok=True)

    return session_id


def get_session_path(session_id):
    return os.path.join(USER_DOCUMENTS_DIR, session_id)

# =====================================================
# METADATA HELPERS
# =====================================================

def load_json(file_path):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def load_pickle(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, "rb") as file:
        return pickle.load(file)

# =====================================================
# DOCUMENT HELPERS
# =====================================================

def get_uploaded_documents(session_id):
    """
    Returns all uploaded documents
    for a particular session.
    """
    metadata_path = os.path.join(get_session_path(session_id), METADATA_FILE)

    metadata = load_json(metadata_path)

    return metadata.get("documents", [])

def get_document_statistics(session_id, document_id):

    metadata_path = os.path.join(get_session_path(session_id), METADATA_FILE)

    metadata = load_json(metadata_path)

    for document in metadata.get("documents", []):
        if document["id"] == document_id:
            return {
                "document_size": document["document_size"],
                "total_characters": document["total_characters"],
                "total_words": document["total_words"],
                "total_chunks": document["total_chunks"],
                "total_embeddings": document["total_embeddings"],
                "embedding_dimension": document["embedding_dimension"],
                "uploaded_at": document["uploaded_at"],
                "expires_at": document["expires_at"]
            }
    return {}


# =====================================================
# CHUNK DETAILS
# =====================================================

def get_chunk_details(session_id, document_id, chunk_number):

    chunk_file = os.path.join(get_session_path(session_id), CHUNK_METADATA_FILE)

    chunks = load_pickle(chunk_file)

    for chunk in chunks:
        if (chunk["document_id"] == document_id and chunk["chunk_number"] == chunk_number):
            return {
                "characters": chunk["character_count"],
                "words": chunk["word_count"],
                "text": chunk["chunk_text"]
            }
    return {}


# =====================================================
# EMBEDDING DETAILS
# =====================================================

def get_embedding_details(session_id, document_id, embedding_number):
    embedding_file = os.path.join(get_session_path(session_id), EMBEDDING_METADATA_FILE)
    embeddings = load_pickle(embedding_file)
    for embedding in embeddings:
        if (embedding["document_id"] == document_id and embedding["embedding_number"] == embedding_number):
            return {
                "dimension": embedding["embedding_dimension"],
                "vector": embedding["embedding_vector"]
            }

    return {}
# =====================================================
# SESSION STATISTICS
# =====================================================

def get_session_statistics(session_id):

    metadata_path = os.path.join(get_session_path(session_id), METADATA_FILE)

    metadata = load_json(metadata_path)

    documents = metadata.get("documents", [])

    total_chunks = 0
    total_embeddings = 0
    total_characters = 0
    total_words = 0

    for document in documents:
        total_chunks += (document["total_chunks"])
        total_embeddings += (document["total_embeddings"])
        total_characters += (document["total_characters"])
        total_words += (document["total_words"])

    return {
        "session_id": session_id,
        "active_documents": len(documents),
        "active_chunks": total_chunks,
        "active_embeddings": total_embeddings,
        "total_characters": total_characters,
        "total_words": total_words
    }

# =====================================================
# DOCUMENT EXPIRY
# =====================================================
def get_remaining_time(document):
    """
    Returns remaining expiry time.
    """
    expiry_time = datetime.strptime(document["expires_at"], "%Y-%m-%d %H:%M:%S")
    remaining = (expiry_time - datetime.now())
    seconds = int(remaining.total_seconds())
    if seconds <= 0:
        return "Expired"
    minutes = seconds // 60
    return f"{minutes} Minutes"

def calculate_expiry_time():
    expiry = datetime.now() + timedelta(seconds=DOCUMENT_TTL_SECONDS)
    return expiry.strftime("%Y-%m-%d %H:%M:%S")


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =====================================================
# DELETE DOCUMENT
# =====================================================

def delete_document(session_id, document_id):
    """
    Deletes a document's metadata.
    NOTE:
    The FAISS index will be rebuilt
    in rag.py after deletion.
    """

    metadata_path = os.path.join(get_session_path(session_id), METADATA_FILE)

    metadata = load_json(metadata_path)

    documents = metadata.get("documents", [])

    updated_documents = []

    for document in documents:
        if document["id"] != document_id:
            updated_documents.append(document)

    metadata["documents"] = (updated_documents)

    save_json(metadata, metadata_path)

# =====================================================
# DOCUMENT EXISTS
# =====================================================
def document_exists(session_id, document_name):

    documents = get_uploaded_documents(session_id)

    for document in documents:
        if (document["name"] == document_name):
            return True

    return False

# =====================================================
# DELETE ENTIRE SESSION
# =====================================================

def delete_session(session_id):

    session_path = get_session_path(session_id)

    if os.path.exists(session_path):
        shutil.rmtree(session_path)

def download_sample_document(file_name):
    path = os.path.join("sample_documents", file_name)

    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    return data