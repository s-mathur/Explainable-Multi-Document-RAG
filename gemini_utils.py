import os
import google.generativeai as genai
import streamlit as st
from config import LLM_MODEL

def build_fallback_response(retrieved_chunks):
    if len(retrieved_chunks) == 0:
        return ("No relevant information was "
                "found in the uploaded documents."
                )
    response = [] 
    response.append("<span style='color:red'>LLM is currently unavailable.</span>\n\n")
    response.append("\nThe following information was "
                    "retrieved from the uploaded documents:\n")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = ( chunk["text"][:300] )
        response.append(
            f"---\n<span><b>Rank {index}</b> | <b>Document</b>: {chunk['document_name']} | <b>Similarity Percentage</b>: {chunk['similarity_percentage']:.2f} %\n\n"
            f"{text}<span>\n\n\n" )
    return "\n".join(response)

# =====================================================
# CONFIGURE GEMINI API
# =====================================================

def configure_gemini_api_key():

    """
    Configures the Gemini API using
    the environment variable.

    For local development:
        export GEMINI_API_KEY=YOUR_API_KEY

    For Hugging Face Spaces:
        Add GEMINI_API_KEY under Secrets.
    """
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Try Hugging Face environment variable.
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    raise ValueError(
        "Gemini API key not found."
    )

    


# =====================================================
# LOAD GEMINI MODEL
# =====================================================

def get_gemini_model():

    """
    Returns the configured Gemini model.
    """

    api_key = configure_gemini_api_key()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=LLM_MODEL)

    return model


# =====================================================
# GENERATE RESPONSE
# =====================================================

def generate_gemini_response(
        question,
        context
):

    """
    Generates the response using Gemini.
    """

    model = get_gemini_model()

    prompt = f"""
    You are a helpful RAG assistant.

    Answer the user's question using the
    provided document context.

    If the answer is not present in the
    provided context, explicitly mention
    that the uploaded documents do not
    contain sufficient information.

    Context:
    {context}

    Question:
    {question}
    """
    try:
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return response.text
        return ("Unable to generate a response from Gemini.")
    except Exception as e:
        raise Exception(str(e))
    