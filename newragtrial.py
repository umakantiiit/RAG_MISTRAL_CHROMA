import streamlit as st
import tempfile
import pathlib

# Import necessary components from LangChain and our custom models
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
import pathlib
import textwrap
from IPython.display import display, Markdown
from google.colab import userdata
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai.chat_models import ChatMistralAI
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint

import pandas as pd

from langchain import PromptTemplate


# Document Loader
from langchain.document_loaders import PyPDFLoader

# Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector DataBase
from langchain.vectorstores import Chroma, FAISS, Pinecone

# Chains - sequence of calls
from langchain.chains import ConversationalRetrievalChain


# --------------------------
# Configuration and Keys
# --------------------------

# Set your API keys (ensure you handle these securely in production!)
GOOGLE_API_KEY = 'AIzaSyBNkK3x3w6H-mBVgFeqL8a2TKTBPejsFvE'
genai.configure(api_key=GOOGLE_API_KEY)
MISTRAL_API_KEY = 'Krlis2XUIQT1qn7IAhJOX8luHjkdMQX2'

# Initialize the Mistral model (you can change model_version if needed)
model = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model_version="mistral-large-latest")

# --------------------------
# Prompt Template
# --------------------------
# This prompt is designed to produce descriptive bullet-point answers.
prompt_template = prompt_template = """
You are an expert medical advisor. Your task is to provide detailed and accurate information about diseases and drugs based on the user's query.If context is provided below, use it to answer the question; otherwise, answer based solely on your general knowledge.


For queries about diseases, provide the following information:
1. The cause of the disease.
2. The main symptoms of the disease.
3. Any available home remedies or Ayurvedic treatments.
4. The potential consequences if the disease is left untreated.
5. The general population affected by the disease.
6. Donot output \n in your answer.

For queries about drugs or their molecules, provide the following information:
1. The mechanism of action of the drug/molecule.
2. The diseases that can be treated using the drug/molecule.
3. The potential effects if a patient skips a dose of the drug.
4. Donot output \n in your answer.

If you are unable to provide accurate information, do not give a wrong answer.Provide all the answer in descriptive manner and in bullet points , so that end user can unsderstand the answer easily.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# --------------------------
# Streamlit App Layout
# --------------------------
st.title("Medical Chatbot (RAG with Mistral)")

st.markdown(
    """
    This app allows you to chat with a medical chatbot. You can either:
    - **Upload a PDF file** (only a medical PDF is allowed) to ask questions based on that document.
    - Or simply ask a general medical question without uploading any file.
    """
)

# File uploader: Accept only PDF files.
uploaded_pdf = st.file_uploader("Upload a Medical PDF", type=["pdf"])

# Input text for the question.
question = st.text_input("Enter your medical question:")

# A button to submit the question.
if st.button("Get Answer"):

    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # --------------------------
        # Case 1: PDF Uploaded
        # --------------------------
        if uploaded_pdf is not None:
            st.info("Processing the PDF and retrieving context...")
            try:
                # Save the uploaded PDF to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    tmp_pdf_path = tmp_file.name

                # Load and split the PDF
                pdf_loader = PyPDFLoader(tmp_pdf_path)
                pages = pdf_loader.load_and_split()

                # Concatenate all page content into one string with double newlines between pages
                full_text = "\n\n".join(page.page_content for page in pages)

                # Split the concatenated text into chunks.
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
                texts = text_splitter.split_text(full_text)

                # Initialize the embeddings and vector store (Chroma in this example)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                vector_index = Chroma.from_texts(texts, embeddings)
                retriever = vector_index.as_retriever()

                # Retrieve relevant documents for the question.
                docs = retriever.get_relevant_documents(question)

                st.markdown("**Retrieved Context:**")
                for i, doc in enumerate(docs):
                    st.write(f"**Document {i+1}:** {doc.page_content[:200]}...")  # Show first 200 chars

                context_text = "\n\n".join(doc.page_content for doc in docs)

                # Build the QA chain with the prompt
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

                # Display the answer in a nicely formatted manner
                st.markdown("### Answer")
                st.markdown(result.get("output_text", "No answer was generated."))
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

        # --------------------------
        # Case 2: No PDF Uploaded
        # --------------------------
        else:
            st.info("No PDF uploaded. Answering based solely on general knowledge...")
            try:
                # When no PDF is provided, we set context as empty.
                empty_context = ""
                # You can choose to call the chain with an empty list of documents,
                # or directly use the model’s chat interface.
                # Here, for consistency, we use the QA chain.
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                result = chain({"input_documents": [], "question": question, "context": empty_context},
                               return_only_outputs=True)

                st.markdown("### Answer")
                st.markdown(result.get("output_text", "No answer was generated."))
            except Exception as e:
                st.error(f"Error generating answer: {e}")
