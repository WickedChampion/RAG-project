"""
RAG Application with Streamlit, Supabase, and LangChain
--------------------------------------------------------
This app allows users to:
1. Upload PDF/TXT documents
2. Index them into Supabase vector store
3. Query the documents using natural language
4. Get AI-generated answers using Groq LLM
"""

import os
from dotenv import load_dotenv
import streamlit as st

from supabase import create_client

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_groq import ChatGroq

from pypdf import PdfReader

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys and URLs from environment
SUPABASE_URL = os.getenv("SUPABASE_URL")  # Supabase project URL
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # Supabase anon/service key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Groq API key for LLM

# Validate required environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Set SUPABASE_URL and SUPABASE_KEY in environment or .env file.")

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# Configure Streamlit page
st.set_page_config(page_title="RAG App with Supabase", layout="wide")

st.title("RAG app")

# -------------------- SIDEBAR: User Authentication --------------------
with st.sidebar:
    st.header("Account")
    email = st.text_input("Email for Supabase login (magic link)")
    if st.button("Send magic link"):
        if not supabase:
            st.error("Supabase client not configured.")
        else:
            try:
                # Send magic link email for passwordless auth
                supabase.auth.sign_in(email=email)
                st.success("Magic link sent to your email (check inbox).")
            except Exception as e:
                st.error(f"Auth error: {e}")

# -------------------- SECTION 1: File Upload & Indexing --------------------
uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"], accept_multiple_files=False)

# Track which files have been indexed (persists during session)
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if uploaded is not None:
    filename = uploaded.name
    st.write(f"Uploaded: {filename}")
    
    # Only show index button if file hasn't been indexed yet
    if filename not in st.session_state.indexed_files:
        if st.button("Index Document"):
            # Extract text based on file type
            if filename.lower().endswith(".pdf"):
                try:
                    # Parse PDF and extract text from all pages
                    reader = PdfReader(uploaded)
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as e:
                    st.error(f"Failed to read PDF: {e}")
                    text = ""
            else:
                # Read plain text file
                text = uploaded.getvalue().decode("utf-8")

            if text.strip():
                # Split text into chunks for embedding
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                
                # Create Document objects with metadata
                docs = [Document(page_content=c, metadata={"source": filename}) for c in chunks]

                st.info(f"Creating embeddings for {len(docs)} chunks...")
                
                # Initialize embedding model (all-MiniLM-L6-v2 produces 384-dim vectors)
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

                if supabase is None:
                    st.error("Supabase not configured. Stop.")
                else:
                    try:
                        # Store document embeddings in Supabase vector table
                        store = SupabaseVectorStore.from_documents(docs, embeddings, client=supabase, table_name="documents")
                        st.session_state.indexed_files.add(filename)
                        st.success("Uploaded and indexed into Supabase vector store (table 'documents').")
                    except Exception as e:
                        st.error(f"Indexing error: {e}")
    else:
        st.success(f"'{filename}' already indexed.")

# -------------------- SECTION 2: Query & Answer --------------------
st.markdown("---")
st.header("Query the indexed documents")
query = st.text_input("Enter your question")

if st.button("Ask"):
    if not query:
        st.error("Enter a query.")
    else:
        if supabase is None:
            st.error("Supabase client not configured.")
        else:
            try:
                # Create embedding for the query
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                query_embedding = embeddings.embed_query(query)
                
                # Perform similarity search using Supabase RPC function
                # match_documents is a custom PostgreSQL function that uses pgvector
                response = supabase.rpc(
                    "match_documents",
                    {"query_embedding": query_embedding, "match_count": 4}
                ).execute()
                
                results = response.data if response.data else []
                
                if not results:
                    st.warning("No matching documents found.")
                elif GROQ_API_KEY:
                    # Use Groq LLM to generate answer from retrieved context
                    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0, api_key=GROQ_API_KEY)
                    
                    # Combine retrieved chunks into context
                    context = "\n\n".join([doc["content"] for doc in results])
                    
                    # Create RAG prompt
                    prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                    
                    # Get LLM response
                    answer = llm.invoke(prompt)
                    st.subheader("Answer")
                    st.write(answer.content)
                else:
                    # Fallback: show raw retrieved chunks if no LLM key
                    st.subheader("Top retrieved chunks")
                    for i, r in enumerate(results, 1):
                        st.write(f"{i}. Source: {r.get('metadata', {}).get('source', 'unknown')}")
                        st.write(r["content"])
            except Exception as e:
                st.error(f"Query error: {e}")
