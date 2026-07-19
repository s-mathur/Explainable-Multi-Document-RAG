import os
import uuid
import pickle
import json
import time
import faiss
import numpy as np
import google.generativeai as genai

from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    CHUNK_METADATA_FILE,
    EMBEDDING_METADATA_FILE
)

from utils import (
    get_session_path,
    load_json,
    save_json,
    load_pickle,
    calculate_expiry_time,
    get_current_time,
    create_user_session
)
from gemini_utils import (generate_gemini_response, build_fallback_response)

# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ==========================================
# TEXT EXTRACTION
# ==========================================

def extract_pdf_text(file):

    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def extract_docx_text(file):
    document = Document(file)
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text

def extract_txt_text(file):
    return file.read().decode("utf-8")

def extract_text(file):
    extension = file.name.split(".")[-1].lower()
    if extension == "pdf":
        return extract_pdf_text(file)
    if extension == "docx":
        return extract_docx_text(file)
    if extension == "txt":
        return extract_txt_text(file)
    raise ValueError(
        "Unsupported file type."
    )

# ==========================================
# CHUNKING
# ==========================================

def split_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_text(text)


# ==========================================
# EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    return embeddings


# ==========================================
# CREATE FAISS INDEX
# ==========================================

def create_faiss_index(embeddings):

    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)

    embeddings = np.array(embeddings, dtype=np.float32)

    index.add(embeddings)

    return index


# ==========================================
# SAVE FAISS INDEX
# ==========================================

def save_faiss_index(index, session_id):
    path = os.path.join(get_session_path(session_id), FAISS_INDEX_FILE)
    faiss.write_index(index, path)


def load_faiss_index(session_id):
    path = os.path.join(get_session_path(session_id), FAISS_INDEX_FILE)
    return faiss.read_index(path)


# ==========================================
# DOCUMENT PROCESSING
# ==========================================

def process_documents(uploaded_files):
    session_id = create_user_session()
    session_path = get_session_path(session_id)
    metadata_path = os.path.join(session_path, METADATA_FILE)
    metadata = load_json(metadata_path)
    documents = metadata.get("documents", [])

    all_chunks = []
    all_embeddings = []

    chunk_metadata = []
    embedding_metadata = []

    existing_chunks = load_pickle(os.path.join(session_path, CHUNK_METADATA_FILE))

    existing_embeddings = load_pickle(
        os.path.join(session_path, EMBEDDING_METADATA_FILE))

    chunk_metadata.extend(existing_chunks)

    embedding_metadata.extend(existing_embeddings)

    for file in uploaded_files:
        document_id = str(uuid.uuid4())
        text = extract_text(file)
        chunks = split_into_chunks(text)
        embeddings = generate_embeddings(chunks)

        documents.append({
            "id": document_id,
            "name": file.name,
            "document_size": file.size,
            "total_characters": len(text),
            "total_words": len(text.split()),
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "uploaded_at": get_current_time(),
            "expires_at": calculate_expiry_time()
        })

        for i, chunk in enumerate(chunks, start=1):
            chunk_metadata.append({
                "document_id": document_id,
                "document_name": file.name,
                "chunk_number": i,
                "chunk_text": chunk,
                "character_count": len(chunk),
                "word_count": len(chunk.split())
            })
            embedding_metadata.append({
                "document_id": document_id,
                "document_name": file.name,
                "embedding_number": i,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "embedding_vector": embeddings[i - 1].tolist()
            })

        all_chunks.extend(chunks)

        all_embeddings.extend(embeddings)

    metadata["documents"] = (documents)

    save_json(metadata, metadata_path)

    with open(os.path.join(session_path, CHUNK_METADATA_FILE), "wb") as file:
        pickle.dump(chunk_metadata, file)

    with open(os.path.join(session_path, EMBEDDING_METADATA_FILE), "wb") as file:

        pickle.dump(embedding_metadata, file)

    combined_embeddings = []

    for item in embedding_metadata:
        combined_embeddings.append(item["embedding_vector"])

    index = create_faiss_index(combined_embeddings)

    save_faiss_index(index, session_id)
    return session_id


# ==========================================
# QUERY EMBEDDING
# ==========================================

def generate_query_embedding(question):
    embedding = embedding_model.encode([question])[0]
    return embedding


# ==========================================
# RETRIEVE TOP K CHUNKS
# ==========================================

def retrieve_chunks(question, session_id):

    query_embedding = (generate_query_embedding(question))

    index = load_faiss_index(session_id)

    distances, indices = index.search(
        np.array([query_embedding], dtype=np.float32), TOP_K)

    chunk_metadata = load_pickle(
        os.path.join(get_session_path(session_id), CHUNK_METADATA_FILE)
    )

    retrieved = []

    for score, idx in zip(distances[0], indices[0]):
        if idx >= len(chunk_metadata):
            continue

        data = chunk_metadata[idx]

        similarity_score = (round(1 / (1 + score), 4))

        retrieved.append({
            "document_name": data["document_name"],
            "chunk_number": data["chunk_number"],
            "embedding_number": data["chunk_number"],
            "similarity_score": similarity_score,
            "similarity_percentage": round(similarity_score * 100, 2),
            "text": data["chunk_text"]
        })

    return (retrieved, query_embedding.tolist())


# ==========================================
# GEMINI RESPONSE
# ==========================================

def generate_response(question, retrieved_chunks):
    gemini_status = "Success"
    gemini_time = 0
    try:
        start_time = time.time()
        context = "\n\n".join(chunk["text"] for chunk in retrieved_chunks)
        answer = generate_gemini_response(question, context)
        gemini_time = round(time.time() - start_time, 4)
    except Exception as e:
        gemini_status = "Failed"
        gemini_time = round(time.time() - start_time, 4)
        answer = build_fallback_response(retrieved_chunks)
    return answer, gemini_status, gemini_time


# ==========================================
# MAIN RESPONSE FUNCTION
# ==========================================

def get_response(question, session_id, selected_documents):

    retrieved_chunks, query_vector = (retrieve_chunks(question, session_id))

    retrieved_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk["document_name"] in selected_documents
    ]

    answer, gemini_status, gemini_time = generate_response(question, retrieved_chunks)

    return {
        "answer": answer,
        "query_statistics": {
            "characters": len(question),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "embedding": query_vector
        },
        "retrieval_statistics": {
            "Total Documents Searched": len(selected_documents),
            "Total Chunks Searched":
            len(
                load_pickle(
                    os.path.join(get_session_path(session_id), CHUNK_METADATA_FILE)
                )
            ),
            "Top K": TOP_K
        },
        "gemini_statistics": {
            "status": gemini_status,
            "response_time": gemini_time
        },
        "source_attribution": list(set([chunk["document_name"] for chunk in retrieved_chunks])),
        "retrieved_chunks": retrieved_chunks
    }
