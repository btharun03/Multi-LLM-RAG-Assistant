# app.py
# Full RAG Chatbot with:
# - Multi-LLM: Gemini (Google Generative API) & HuggingFace (LLaMA3)
# - RAG: LangChain + Chroma persistent vector store
# - HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
# - Advanced memory: buffer + summary + vector chat recall
# - Streaming responses for Gemini when available
# - Source highlighting, confidence scores, export PDF, multi-file ingestion

import os
import io
import shutil
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional, Any, Iterable, Dict, Union, cast

import streamlit as st

# Avoid TensorFlow accidental import via transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# LangChain + community imports (current recommended paths)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub

# Embeddings package bridge (requires langchain-huggingface)
# and sentence-transformers installed locally
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None  # handled later

# Memory
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# Optional Gemini wrapper
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# Document loaders (community)
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    BSHTMLLoader,
)

# Optional PDF export
try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="RAG Chatbot • LLaMA3 & Gemini", layout="wide")
st.title("💬 RAG Chatbot — Multi-LLM (Gemini / LLaMA3) + RAG + Advanced Memory")
st.write(
    "Upload documents (pdf/txt/docx/csv/html), build a persistent KB, then chat. "
    "Switch between Gemini (Google) and HuggingFace (LLaMA3) backends."
)

# ----------------------------
# Session-state defaults
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Tuple[str, str]] = []
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = "chroma_store"
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "k_chunks" not in st.session_state:
    st.session_state.k_chunks = 4
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "vector_chat_store" not in st.session_state:
    st.session_state.vector_chat_store = None
if "summary_memory" not in st.session_state:
    st.session_state.summary_memory = None

# ----------------------------
# Helper: initialize embeddings robustly
# ----------------------------
def init_local_embeddings() -> Optional[Any]:
    """Initialize local HuggingFace embeddings using sentence-transformers (preferred)."""
    if HuggingFaceEmbeddings is None:
        return None
    try:
        emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return emb
    except Exception as e:
        st.warning(f"Local HF embeddings failed: {e}")
        return None

def init_remote_hf_embeddings() -> Optional[Any]:
    """Attempt to use HuggingFace Hub inference embeddings if token available."""
    try:
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if hf_token:
            # langchain-community fallback: HuggingFaceHubEmbeddings may exist
            try:
                from langchain_community.embeddings import HuggingFaceHubEmbeddings
                return HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                return None
    except Exception:
        return None

def init_gemini_embeddings() -> Optional[Any]:
    """Try Google Generative AI embeddings if GOOGLE_API_KEY available."""
    if not HAS_GEMINI:
        return None
    try:
        if os.getenv("GOOGLE_API_KEY"):
            # class name depends on package version
            try:
                return GoogleGenerativeAIEmbeddings(model="models/text-embedding-3-large")
            except Exception:
                # fallback to other supported model names if needed
                return GoogleGenerativeAIEmbeddings(model="models/text-embedding-3-small")
    except Exception:
        return None

def init_embeddings() -> Any:
    # Try local (fast, offline)
    emb = init_local_embeddings()
    if emb:
        return emb
    # Try HF inference
    emb = init_remote_hf_embeddings()
    if emb:
        return emb
    # Try Gemini embeddings
    emb = init_gemini_embeddings()
    if emb:
        return emb
    return None

# Initialize embeddings (only once)
if st.session_state.embeddings is None:
    st.session_state.embeddings = init_embeddings()
    if st.session_state.embeddings is None:
        st.error(
            "❌ Could not initialize any embeddings backend.\n\n"
            "Fix local install: `pip install -U sentence-transformers torch langchain-huggingface`\n"
            "Or set environment variables for a remote provider (HUGGINGFACEHUB_API_TOKEN or GOOGLE_API_KEY)."
        )
        st.stop()

# ----------------------------
# Vector store (persistent) builder
# ----------------------------
@st.cache_resource(show_spinner=True)
def build_vectorstore(persist_directory: str) -> Chroma:
    os.makedirs(persist_directory, exist_ok=True)
    vs = Chroma(
        collection_name="knowledge_base",
        persist_directory=persist_directory,
        embedding_function=st.session_state.embeddings,
    )
    return vs

@st.cache_resource(show_spinner=True)
def get_vector_chat_store(persist_directory: str) -> Chroma:
    chatdir = os.path.join(persist_directory, "chat_memory")
    os.makedirs(chatdir, exist_ok=True)
    vs = Chroma(
        collection_name="chat_mem",
        persist_directory=chatdir,
        embedding_function=st.session_state.embeddings,
    )
    return vs

# ensure vector stores exist in session_state
st.session_state.vectorstore = build_vectorstore(st.session_state.persist_dir)
st.session_state.vector_chat_store = get_vector_chat_store(st.session_state.persist_dir)

# ----------------------------
# Sidebar: KB controls & uploads
# ----------------------------
st.sidebar.header("Knowledge Base")
st.session_state.persist_dir = st.sidebar.text_input("Chroma persist directory", value=st.session_state.persist_dir)
st.session_state.k_chunks = st.sidebar.slider("Top-K chunks", 2, 10, st.session_state.k_chunks)
st.session_state.chunk_size = st.sidebar.slider("Chunk size", 500, 2000, st.session_state.chunk_size, step=100)
st.session_state.chunk_overlap = st.sidebar.slider("Chunk overlap", 50, 400, st.session_state.chunk_overlap, step=50)

uploads = st.sidebar.file_uploader(
    "Upload documents (TXT / PDF / DOCX / CSV / HTML)",
    type=["txt", "pdf", "docx", "csv", "htm", "html"],
    accept_multiple_files=True,
)

colA, colB = st.sidebar.columns(2)
clear_kb = colA.button("🧹 Clear KB")
rebuild_index = colB.button("🔁 Rebuild Index")

if clear_kb:
    try:
        if os.path.isdir(st.session_state.persist_dir):
            shutil.rmtree(st.session_state.persist_dir)
        os.makedirs(st.session_state.persist_dir, exist_ok=True)
        st.session_state.vectorstore = build_vectorstore(st.session_state.persist_dir)
        st.session_state.vector_chat_store = get_vector_chat_store(st.session_state.persist_dir)
        st.sidebar.success("Knowledge base cleared.")
    except Exception as e:
        st.sidebar.error(f"Failed to clear KB: {e}")

# Handle uploads
if uploads:
    tmpdir = tempfile.mkdtemp()
    all_docs: List[Document] = []
    try:
        for f in uploads:
            path = os.path.join(tmpdir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            ext = os.path.splitext(path)[1].lower()
            if ext == ".txt":
                all_docs.extend(TextLoader(path, encoding="utf-8").load())
            elif ext == ".pdf" and PyPDFLoader is not None:
                all_docs.extend(PyPDFLoader(path).load())
            elif ext == ".docx" and Docx2txtLoader is not None:
                all_docs.extend(Docx2txtLoader(path).load())
            elif ext == ".csv" and CSVLoader is not None:
                all_docs.extend(CSVLoader(path).load())
            elif ext in (".htm", ".html") and BSHTMLLoader is not None:
                all_docs.extend(BSHTMLLoader(path).load())
            else:
                st.warning(f"No loader for file {f.name}. It may be unsupported or optional deps are missing.")
        if all_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
            chunks = splitter.split_documents(all_docs)
            st.session_state.vectorstore.add_documents(chunks)
            st.session_state.vectorstore.persist()
            st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(uploads)} files.")
    except Exception as e:
        st.sidebar.error(f"Failed to process uploaded files: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if rebuild_index:
    try:
        st.session_state.vectorstore.persist()
        st.sidebar.success("Index persisted / rebuilt.")
    except Exception as e:
        st.sidebar.error(f"Failed to rebuild index: {e}")

# ----------------------------
# LLM selection + builder
# ----------------------------
st.sidebar.header("LLM / Model")
llm_options = ["Gemini (Google)"] if HAS_GEMINI else []
llm_options += ["LLaMA 3 (HuggingFace)"]
llm_choice = st.sidebar.selectbox("Choose backend LLM", options=llm_options, index=0 if HAS_GEMINI else 0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
streaming_enabled = st.sidebar.checkbox("Streaming (Gemini)", value=True)

# Build LLM cached factory
@st.cache_resource
def get_llm_cached(choice: str, temp: float):
    if choice.startswith("Gemini"):
        if not HAS_GEMINI:
            raise RuntimeError("Gemini wrapper package not installed (langchain_google_genai).")
        # ChatGoogleGenerativeAI uses GOOGLE_API_KEY env var or accepted param depending on version
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temp, streaming=streaming_enabled)
    else:
        # HuggingFace LLM via HF Hub (requires HUGGINGFACEHUB_API_TOKEN env var)
        return HuggingFaceHub(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", model_kwargs={"temperature": temp, "max_new_tokens": 512})

# Initialize memory components
if st.session_state.summary_memory is None:
    try:
        st.session_state.summary_memory = ConversationSummaryMemory(llm=get_llm_cached(llm_choice, 0.0), memory_key="summary", return_messages=True)
    except Exception:
        # fallback to buffer if summary memory init fails
        st.session_state.summary_memory = None

buffer_memory = ConversationBufferMemory(memory_key="chat_buffer", return_messages=True)

# Ensure vector chat store present (already done earlier)
vector_chat_store = st.session_state.vector_chat_store

# ----------------------------
# Prompt template to combine context + memory
# ----------------------------
ANSWER_PROMPT = PromptTemplate(
    input_variables=["summary", "recent", "recalled", "context", "question"],
    template=(
        "You are a helpful assistant. Use the summary of previous conversation, recent turns, and retrieved context to answer.\n\n"
        "SUMMARY:\n{summary}\n\n"
        "RECENT:\n{recent}\n\n"
        "RECALLED_SIMILAR_CHATS:\n{recalled}\n\n"
        "CONTEXT:\n{context}\n\n"
        "Question: {question}\nAnswer concisely, and cite sources in the form [S1], [S2] when used.\n"
    )
)

# ----------------------------
# Utility functions: retrieve, recall, upsert
# ----------------------------
def retrieve_with_scores(query: str, k: int) -> List[Tuple[Document, float]]:
    """Return list of (Document, score). If backend supports scores, use them."""
    try:
        docs_and_scores = st.session_state.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        # returns list of tuples (Document, score)
        return docs_and_scores
    except Exception:
        docs = st.session_state.vectorstore.similarity_search(query, k=k)
        return [(d, 0.0) for d in docs]

def recall_similar_chats(query: str, k: int = 3) -> List[Document]:
    try:
        docs_and_scores = st.session_state.vector_chat_store.similarity_search_with_relevance_scores(query, k=k)
        return [d for (d, s) in docs_and_scores]
    except Exception:
        return st.session_state.vector_chat_store.similarity_search(query, k=k)

def upsert_chat_turn_to_vector(user_q: str, bot_a: str):
    texts = [f"User: {user_q}", f"Bot: {bot_a}"]
    docs = [Document(page_content=t, metadata={"type": "chat", "ts": datetime.utcnow().isoformat()}) for t in texts]
    st.session_state.vector_chat_store.add_documents(docs)
    st.session_state.vector_chat_store.persist()

# ----------------------------
# Chat UI + Logic
# ----------------------------
st.subheader("Chat with your knowledge base")
query = st.text_input("Ask a question (press Enter or click Ask):")

# show existing chat
for speaker, text in st.session_state.chat_history:
    if speaker == "user":
        st.markdown(f"🧑 **You:** {text}")
    else:
        st.markdown(f"🤖 **Bot:** {text}")

if query:
    # 1) Retrieve context with scores
    retrieved = retrieve_with_scores(query, st.session_state.k_chunks)
    context_blocks = []
    sources_meta = []
    for idx, (doc, score) in enumerate(retrieved, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        context_blocks.append(f"[S{idx}] {snippet}")
        sources_meta.append({"id": idx, "score": float(score) if score is not None else 0.0, "source": doc.metadata.get("source", "unknown"), "snippet": snippet[:500]})

    # 2) Recall similar past chats from vector chat memory
    recalled_docs = recall_similar_chats(query, k=3)
    recalled_text = "\n".join([d.page_content for d in recalled_docs]) if recalled_docs else "(none)"

    # 3) Recent chat buffer (last few turns)
    recent_turns = st.session_state.chat_history[-6:]
    recent_text = "\n".join([f"{s.upper()}: {m}" for s, m in recent_turns]) if recent_turns else "(none)"

    # 4) Summary memory (try to update)
    summary_text = "(no summary)"
    try:
        if st.session_state.summary_memory is not None:
            # add the user query to memory and then load variables
            st.session_state.summary_memory.chat_memory.add_user_message(query)
            summary_text = st.session_state.summary_memory.load_memory_variables({}).get("history", "")
    except Exception:
        summary_text = "(summary unavailable)"

    # 5) Compose prompt
    context_text = "\n".join(context_blocks) if context_blocks else "(no context found)"
    prompt_text = ANSWER_PROMPT.format(
        summary=summary_text,
        recent=recent_text,
        recalled=recalled_text,
        context=context_text,
        question=query,
    )

    # 6) Build LLM and run (stream if Gemini & streaming enabled)
    llm = get_llm_cached(llm_choice, temperature)
    bot_answer = ""
    stream_placeholder = None
    try:
        if HAS_GEMINI and isinstance(llm, ChatGoogleGenerativeAI) and streaming_enabled:
            # stream tokens (API wrapper must support .stream or similar)
            stream_placeholder = st.empty()
            for chunk in llm.stream(prompt_text):
                # chunk may vary by wrapper; try to read content/text
                token = getattr(chunk, "content", None) or getattr(chunk, "text", None) or ""
                bot_answer += token
                stream_placeholder.markdown(f"🤖 **Bot:** {bot_answer}")
            if not bot_answer:
                # fallback
                resp = llm.invoke(prompt_text)
                bot_answer = getattr(resp, "content", resp)
                if stream_placeholder:
                    stream_placeholder.markdown(f"🤖 **Bot:** {bot_answer}")
        else:
            # Non-streaming: try predict/invoke
            try:
                # Some LLMs have .predict
                bot_answer = llm.predict(prompt_text)
            except Exception:
                resp = llm.invoke(prompt_text)
                bot_answer = getattr(resp, "content", resp)
            st.markdown(f"🤖 **Bot:** {bot_answer}")
    except Exception as e:
        st.error(f"LLM failed: {e}")
        bot_answer = f"Error generating answer: {e}"

    # 7) Save conversation (buffer, vector memory, summary memory)
    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", bot_answer))
    # vector store chat memory
    try:
        upsert_chat_turn_to_vector(query, bot_answer)
    except Exception:
        st.warning("Failed to upsert chat to vector memory.")

    try:
        if st.session_state.summary_memory is not None:
            st.session_state.summary_memory.chat_memory.add_ai_message(bot_answer)
    except Exception:
        pass

    # 8) Show sources and confidence
    with st.expander("📂 Sources & Confidence"):
        if not sources_meta:
            st.write("No sources retrieved.")
        for s in sources_meta:
            st.markdown(f"**[S{s['id']}]** Relevance score: `{s['score']:.3f}` — Source: {s['source']}")
            st.write("> " + s['snippet'] + " ...")

# ----------------------------
# Export conversation as PDF / Download
# ----------------------------
st.markdown("---")
st.subheader("Export Conversation")
if HAS_FPDF:
    if st.button("📄 Export chat to PDF"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="RAG Chatbot Transcript", ln=True)
        pdf.ln(6)
        for speaker, msg in st.session_state.chat_history:
            prefix = "You: " if speaker == "user" else "Bot: "
            for line in (prefix + msg).split("\n"):
                pdf.multi_cell(0, 8, txt=line)
            pdf.ln(2)
        out_fname = f"chat_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
        out_path = os.path.join(tempfile.gettempdir(), out_fname)
        pdf.output(out_path)
        with open(out_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=out_fname, mime="application/pdf")
else:
    st.info("Install fpdf to enable PDF export: `pip install fpdf`")

# ----------------------------
# Diagnostics panel
# ----------------------------
with st.expander("🔎 Diagnostics"):
    import sys, platform
    st.write("Python:", sys.version)
    st.write("Platform:", platform.platform())
    st.write("Streamlit exec:", sys.executable)
    try:
        import sentence_transformers, transformers, torch
        st.write("sentence-transformers:", sentence_transformers.__version__, sentence_transformers.__file__)
        st.write("transformers:", transformers.__version__)
        st.write("torch:", torch.__version__)
    except Exception as e:
        st.write("Sentence-transformers/transformers/torch check failed:", e)
    # show env keys presence
    st.write("HUGGINGFACEHUB_API_TOKEN present:", bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")))
    st.write("GOOGLE_API_KEY present:", bool(os.getenv("GOOGLE_API_KEY")))

st.caption("Tip: set HUGGINGFACEHUB_API_TOKEN and/or GOOGLE_API_KEY in your environment for respective providers.")
