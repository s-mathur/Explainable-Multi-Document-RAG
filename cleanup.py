import os
import pickle
import shutil
from datetime import datetime
from datetime import datetime
from config import (
    USER_DOCUMENTS_DIR,
    METADATA_FILE,
    CHUNK_METADATA_FILE,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE
)

from utils import (
    load_json,
    save_json,
    delete_session,
    get_uploaded_documents,
    delete_document
)

import faiss
import numpy as np

def are_documents_expired(session_id):
    """ Checks whether any uploaded document has expired. 
    If expired, deletes the document metadata and returns True. 
    """
    documents = get_uploaded_documents(session_id)
    if len(documents) == 0:
        return False
    expired_found = False
    current_time = datetime.now()
    for document in documents:
        expiry_time = datetime.strptime(
            document["expires_at"],
            "%Y-%m-%d %H:%M:%S"
        ) 
        if current_time > expiry_time:
            delete_document(
                session_id=session_id,
                document_id=document["id"]
            ) 
            expired_found = True 
    return expired_found
# =====================================================
# DELETE FAISS INDEX
# =====================================================

def delete_faiss_index(session_path):

    index_path = os.path.join(session_path, FAISS_INDEX_FILE)

    if os.path.exists(index_path):
        os.remove(index_path)


# =====================================================
# REBUILD FAISS INDEX
# =====================================================

def rebuild_faiss_index(session_path, embeddings):

    if len(embeddings) == 0:
        delete_faiss_index(session_path)
        return

    embeddings = np.array(embeddings, dtype=np.float32)

    embedding_dimension = (embeddings.shape[1])

    index = faiss.IndexFlatL2(embedding_dimension)

    index.add(embeddings)

    faiss.write_index(index, os.path.join(session_path, FAISS_INDEX_FILE))


# =====================================================
# REMOVE EXPIRED DOCUMENTS
# =====================================================

def remove_expired_documents(session_path):
    metadata_path = os.path.join(session_path, METADATA_FILE)
    if not os.path.exists(metadata_path):
        shutil.rmtree(session_path)
        return
    metadata = load_json(metadata_path)
    documents = metadata.get("documents", [])
    if not documents:
        shutil.rmtree(session_path)
        return
    valid_documents = []
    expired_document_ids = []
    current_time = (datetime.now())
    # -----------------------------------------
    # CHECK EXPIRY
    # -----------------------------------------
    for document in documents:
        expiry_time = datetime.strptime(document["expires_at"], "%Y-%m-%d %H:%M:%S")
        if current_time > expiry_time:
            expired_document_ids.append(document["id"])
        else:
            valid_documents.append(document)

    # No expired documents.
    if len(expired_document_ids) == 0:
        return
    # -----------------------------------------
    # DELETE ENTIRE SESSION
    # -----------------------------------------
    if len(valid_documents) == 0:
        session_id = os.path.basename(session_path)
        delete_session(session_id)
        return
    # -----------------------------------------
    # UPDATE DOCUMENT METADATA
    # -----------------------------------------

    metadata["documents"] = (valid_documents)
    save_json(metadata, metadata_path)

    # -----------------------------------------
    # FILTER CHUNK METADATA
    # -----------------------------------------

    chunk_file = os.path.join(session_path, CHUNK_METADATA_FILE
    )
    if os.path.exists(chunk_file):
        with open(chunk_file, "rb") as file:
            chunks = pickle.load(file)
    else:
        chunks = []

    updated_chunks = [
        chunk for chunk in chunks if chunk["document_id"] not in expired_document_ids
    ]
    with open(chunk_file, "wb") as file:
        pickle.dump(updated_chunks, file)

    # -----------------------------------------
    # FILTER EMBEDDING METADATA
    # -----------------------------------------

    embedding_file = os.path.join(session_path, EMBEDDING_METADATA_FILE)

    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as file:
            embeddings = pickle.load(file)
    else:
        embeddings = []

    updated_embeddings = [
        embedding for embedding in embeddings if embedding["document_id"] not in expired_document_ids
    ]

    with open(embedding_file, "wb") as file:
        pickle.dump(updated_embeddings, file)

    # -----------------------------------------
    # REBUILD FAISS INDEX
    # -----------------------------------------

    vectors = [item["embedding_vector"] for item in updated_embeddings]
    rebuild_faiss_index(session_path, vectors)


# =====================================================
# CLEANUP ALL SESSIONS
# =====================================================

def cleanup_expired_documents():
    if not os.path.exists(USER_DOCUMENTS_DIR):
        return
    sessions = os.listdir(USER_DOCUMENTS_DIR)
    for session_id in sessions:
        session_path = os.path.join(USER_DOCUMENTS_DIR, session_id)
        if os.path.isdir(session_path):
            remove_expired_documents(session_path)
