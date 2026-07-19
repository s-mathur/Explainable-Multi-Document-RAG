# app.py

import streamlit as st

from cleanup import cleanup_expired_documents, are_documents_expired

from utils import (
    get_session_statistics, get_uploaded_documents,
    get_document_statistics, get_chunk_details, get_embedding_details,
    get_remaining_time, delete_document,download_sample_document
)

from rag import (process_documents, get_response)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(page_title="RAG System", layout="wide")

# =====================================================
# SESSION INITIALIZATION
# =====================================================

cleanup_expired_documents()

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "response" not in st.session_state:
    st.session_state.response = None

# =====================================================
# TITLE
# =====================================================

st.markdown(f"<h3>RAG System</h3>", unsafe_allow_html=True)
st.caption("**The data will be sent to the LLM model. Please upload dummy or sample data. The uploaded document will be removed after 30 min automatically.**")

left, middle, right = st.columns(
    [2, 5, 2]
)

# =====================================================
# DOCUMENT UPLOAD
# =====================================================
with left:
    st.write("**Upload Documents**")
    uploaded_files = st.file_uploader(
        label="Upload PDF, TXT or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                session_id = process_documents(uploaded_files=uploaded_files)
                st.session_state.session_id = session_id
            st.success("Documents processed successfully.")
            st.rerun()

    # =====================================================
    # UPLOADED DOCUMENTS
    # =====================================================
    if st.session_state.session_id:
        st.write("**Uploaded Documents**")
        documents = get_uploaded_documents(st.session_state.session_id)
        if len(documents) == 0:
            st.info("No documents uploaded.")
        else:
            with st.expander("Show Uploaded Documents"):
                for document in documents:
                    st.subheader(document["name"])
                    doc_stats = get_document_statistics(st.session_state.session_id, document["id"])
                    st.write(f"Document Size : {doc_stats['document_size']}")
                    st.write(f"Total Characters : {doc_stats['total_characters']}")
                    st.write(f"Total Words : {doc_stats['total_words']}")
                    st.write(f"Total Chunks : {doc_stats['total_chunks']}")
                    st.write(f"Total Embeddings : {doc_stats['total_embeddings']}")
                    st.write(f"Embedding Dimension : {doc_stats['embedding_dimension']}")
                    st.write(f"Uploaded At : {doc_stats['uploaded_at']}")
                    st.write(f"Expires At : {doc_stats['expires_at']}")
                    st.write(f"Time Remaining : {get_remaining_time(document)}")

                    # -----------------------------------------
                    # DELETE BUTTON
                    # -----------------------------------------

                    if st.button(f"Delete {document['name']}", key=document["id"]
                    ):
                        delete_document(session_id=st.session_state.session_id, document_id=document["id"])
                        st.success(f"{document['name']} deleted.")
                        st.rerun()

                    # -----------------------------------------
                    # CHUNK DETAILS
                    # -----------------------------------------

                    with st.expander(f"Chunk Details - {document['name']}"):
                        total_chunks = doc_stats["total_chunks"]
                        if total_chunks > 0:
                            chunk_number = st.slider(
                                label=f"Chunk Number ({document['name']})",
                                min_value=1,
                                max_value=total_chunks,
                                value=1,
                                key=f"chunk_{document['id']}"
                            )

                            chunk = get_chunk_details(
                                session_id=st.session_state.session_id,
                                document_id=document["id"],
                                chunk_number=chunk_number
                            )
                            st.write(f"Chunk Number : {chunk_number}")
                            st.write(f"Characters : {chunk['characters']}")
                            st.write(f"Words : {chunk['words']}")
                            if st.checkbox("Show Chunk Text", key=f"show_chunk_{document['id']}"):
                                st.text(chunk["text"])

                    # -----------------------------------------
                    # EMBEDDING DETAILS
                    # -----------------------------------------

                    with st.expander(f"Embedding Details - {document['name']}"):
                        total_embeddings = (doc_stats["total_embeddings"])
                        if total_embeddings > 0:
                            embedding_number = st.slider(
                                label=(f"Embedding Number ({document['name']})"),
                                min_value=1,
                                max_value=total_embeddings,
                                value=1,
                                key=f"embed_{document['id']}"
                            )
                            embedding = get_embedding_details(
                                session_id=st.session_state.session_id,
                                document_id=document["id"],
                                embedding_number=embedding_number
                            )
                            st.write(f"Embedding Number : {embedding_number}")
                            st.write(f"Embedding Dimension : {embedding['dimension']}")
                            if st.checkbox("Show Embedding Vector", key=f"vector_{document['id']}"):
                                st.write(embedding["vector"])


        # =====================================================
        # SEARCH SCOPE
        # =====================================================

        st.write("**Search Scope**")

        selected_documents = []
        search_scope = st.radio(
            label="Search Scope",
            options=[
                "All Documents",
                "Selected Documents"
            ],
            index=0
        )

        if search_scope == "All Documents":
            selected_documents = [doc["name"] for doc in documents]
        else:
            selected_documents = st.multiselect(
                "Select Documents",
                options=[doc["name"] for doc in documents]
            )


# =====================================================
# ASK QUESTION
# =====================================================
with middle:
    st.write("**Ask Question**")

    question = st.text_area(label="Enter your question", height=120)

    # =====================================================
    # GENERATE RESPONSE
    # =====================================================

    if st.button("Ask"):
        if not st.session_state.session_id:
            st.error(f"No Documents Uploaded.")
        elif question.strip() == "":
            st.error("Please enter a question.")
        elif st.session_state.session_id and are_documents_expired(st.session_state.session_id):
            st.error("The uploaded documents have expired after 30 minutes.\n\n"
                    "Please upload the document(s) again to continue.")
        else:    
            with st.spinner("Generating response..."):
                st.session_state.response = get_response(
                    question=question,
                    session_id=st.session_state.session_id,
                    selected_documents=selected_documents
                )
    if st.session_state.response:
        # ================================================
        # GEMINI RESPONSE
        # ================================================
        stats = st.session_state.response["gemini_statistics"]
        
        if stats['status'] == "Failed":
            st.markdown(st.session_state.response["answer"], unsafe_allow_html=True)
        else:
            st.success(f"{st.session_state.response["answer"]}")
            # st.success(st.session_state.response["answer"])
        # ================================================
        # SOURCE ATTRIBUTION
        # ================================================
        st.divider()
        st.write("**Sources**")
        for source in st.session_state.response["source_attribution"]:
            st.write(source)

        # ================================================
        # RETRIEVED CHUNK DETAILS
        # ================================================

        with st.expander("Retrieved Chunk Details"):
            retrieved_chunks = st.session_state.response["retrieved_chunks"]
            if len(retrieved_chunks) > 0:
                chunk_number = st.slider(
                    label="Retrieved Chunk Number",
                    min_value=1,
                    max_value=len(retrieved_chunks),
                    value=1,
                    key="retrieved_chunk_slider"
                )
                chunk = retrieved_chunks[chunk_number - 1]
                st.write(f"Document Name : {chunk['document_name']}")
                st.write(f"Chunk Number : {chunk['chunk_number']}")
                st.write(f"Embedding Number : {chunk['embedding_number']}")
                st.write(f"Similarity Score : {chunk['similarity_score']}")
                st.write(f"Similarity Percentage : {chunk['similarity_percentage']}")
                if st.checkbox("Show Retrieved Chunk Text", key=f"retrieved_chunk_text_{chunk_number}"):
                    st.text(chunk["text"])

with right:
    if st.session_state.session_id:
        session_stats = get_session_statistics(st.session_state.session_id)

        # =====================================================
        # RAG STATISTICS
        # =====================================================

        st.write("**RAG Statistics**")
        with st.expander("Show RAG Statistics"):
            st.write(f"Total Documents : {session_stats['active_documents']}")
            st.write(f"Total Chunks : {session_stats['active_chunks']}")
            st.write(f"Total Embeddings : {session_stats['active_embeddings']}")
            st.write(f"Total Characters : {session_stats['total_characters']}")
            st.write(f"Total Words : {session_stats['total_words']}")

    if st.session_state.response:
        st.write("**Query Statistics**")
        with st.expander("Show Query Statistics"):
            query_stats = st.session_state.response["query_statistics"]
            st.write(f"Question : {question}")
            st.write(f"Characters : {query_stats['characters']}")
            st.write(f"Embedding Dimension : {query_stats['embedding_dimension']}")
            if st.checkbox("Show Query Embedding"):
                st.write(query_stats["embedding"])

        # ================================================
        # RETRIEVAL STATISTICS
        # ================================================

        st.write("**Retrieval Statistics**")
        with st.expander("Show Retrieval Statistics"):
            retrieval = st.session_state.response["retrieval_statistics"]

            for key, value in retrieval.items():
                st.write(f"{key} : {value}")
    
    st.write("**Sample Documents and Questions**")
    st.download_button(
        label="Health Insurance Policy",
        data=download_sample_document("Health_Insurance_Policy.txt"),
        file_name="Health_Insurance_Policy.txt",
        mime="text/plain"
    )
    st.download_button(
        label="Insurance Claim Guidelines",
        data=download_sample_document("Insurance_Claim_Guidelines.txt"),
        file_name="Insurance_Claim_Guidelines.txt",
        mime="text/plain"
    )
    st.download_button(
        label="Travel Insurance Policy",
        data=download_sample_document("Travel_Insurance_Policy.txt"),
        file_name="Travel_Insurance_Policy.txt",
        mime="text/plain"
    )
    with st.expander("Health Insurance Policy Questions"):
        st.code(download_sample_document("Health_Insurance_Policy_Ques.txt"))
    with st.expander("Insurance Claim Guidelines Questions"):
        st.code(download_sample_document("Insurance_Claim_Guidelines_Ques.txt"))
    with st.expander("Travel Insurance Policy Questions"):
        st.code(download_sample_document("Travel_Insurance_Policy_Ques.txt"))
