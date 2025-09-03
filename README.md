# Multi-LLM RAG Assistant

This repository contains a powerful and feature-rich Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, and LangChain. It allows you to chat with your own documents, leveraging state-of-the-art LLMs like Google's Gemini and Meta's LLaMA 3, combined with a sophisticated multi-layered memory system.

---

## ✨ Features

-   **Multi-LLM Support**: Seamlessly switch between **Google Gemini** (via Google Generative AI API) and **Meta LLaMA 3** (via HuggingFace Hub).
-   **Advanced RAG Pipeline**:
    -   Ingest multiple document types (`.pdf`, `.txt`, `.docx`, `.csv`, `.html`).
    -   Uses **ChromaDB** for a persistent, on-disk vector store.
    -   Configurable chunking strategy and retriever settings (`top-k`).
-   **Flexible Embedding Models**: Automatically selects the best available embedding model:
    1.  **Local**: `sentence-transformers/all-MiniLM-L6-v2` (fast, free, offline).
    2.  **Remote**: HuggingFace Hub or Google Generative AI embeddings (if API keys are present).
-   **Sophisticated Multi-Layered Memory**:
    -   **📝 Buffer Memory**: Remembers the most recent turns of the conversation.
    -   **🧠 Summary Memory**: Creates a running summary of the entire conversation to maintain long-term context.
    -   **💾 Vector Recall Memory**: Embeds and stores every chat turn in a separate vector store, allowing the bot to recall similar past interactions.
-   **Modern User Interface (Streamlit)**:
    -   **Streaming Responses**: Real-time token streaming for a smooth chat experience (Gemini).
    -   **Source Highlighting**: Displays the exact text chunks and relevance scores used to generate an answer.
    -   **PDF Chat Export**: Download your entire conversation history as a PDF.
    -   **Diagnostics Panel**: Easily check your environment and library versions.

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Prerequisites

-   Python 3.12

### 2. Clone the Repository

```bash
git clone [https://github.com/btharun03/Multi-LLM-RAG-Assistant.git](https://github.com/btharun03/Multi-LLM-RAG-Assistant.git)
cd Multii-LLM-RAG-Assistant
```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

<details>
<summary>Or, create a <code>requirements.txt</code> file with the following content:</summary>

```txt
streamlit
langchain
langchain-community
langchain-text-splitters
chromadb
langchain-huggingface
sentence-transformers
torch
langchain-google-genai
pypdf
docx2txt
beautifulsoup4
fpdf
python-dotenv
```

</details>

### 4. Configure API Keys

To use the remote LLMs (Gemini, LLaMA 3) and remote embedding models, you need to set up API keys. The recommended way is to create a `.env` file in the root of your project.

1.  Create a file named `.env`:
    ```
    touch .env
    ```
2.  Add your API keys to the file:
    ```env
    # Get from [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

    # Get from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
    HUGGINGFACEHUB_API_TOKEN="YOUR_HUGGINGFACE_API_TOKEN"
    ```

The application will automatically load these keys. If you don't provide any keys, the app will be limited to using local sentence-transformer embeddings and won't be able to use the LLMs.

### 6. Run the Application

Launch the Streamlit app with the following command:

```bash
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

---

## 📖 How to Use

1.  **Upload Documents**: Use the sidebar to upload one or more documents. The app will automatically process, chunk, and index them into the Chroma vector store.
2.  **Configure Settings**: Adjust the RAG parameters like chunk size, overlap, and the number of chunks (`Top-K`) to retrieve.
3.  **Select an LLM**: Choose between Gemini and LLaMA 3 from the sidebar. You can also adjust the `temperature` for more creative or deterministic responses.
4.  **Start Chatting**: Type your question in the input box at the bottom and press Enter.
5.  **View Sources**: After receiving a response, expand the "Sources & Confidence" section to see which parts of your documents were used to generate the answer.
6.  **Export Chat**: Click the "Export chat to PDF" button to save your session.

---

## 🔧 Code Overview

-   `app.py`: The main Streamlit application file containing all the logic.
-   **Initialization**: Sets up the Streamlit page, session state defaults, and robustly initializes embedding models.
-   **Vector Store**: Uses `@st.cache_resource` to create and manage persistent ChromaDB vector stores for both the knowledge base and the chat memory.
-   **Sidebar Controls**: All user-configurable options (file uploads, RAG settings, LLM selection) are located in the sidebar.
-   **RAG & Memory Logic**:
    -   When a query is submitted, the app first retrieves context from the document vector store (`retrieve_with_scores`).
    -   It then recalls similar past conversations from the chat memory vector store (`recall_similar_chats`).
    -   It fetches recent history from the buffer and a long-term summary from the summary memory.
    -   All these context pieces are formatted into a detailed prompt (`ANSWER_PROMPT`).
-   **LLM Invocation**: The chosen LLM is invoked with the composed prompt. Streaming is enabled for Gemini to provide a better user experience.
-   **State Management**: The chat history and memory objects are updated and persisted after each turn.

---

## 💡 Future Enhancements

This project has a solid foundation that can be extended with many exciting features. Here is a potential roadmap:

-   [ ] **Agent Tools Integration**: Equip the LLM with tools (e.g., web search, calculator, code interpreter) to answer questions that require real-time information or computation.
-   [ ] **Expanded LLM & Vector Store Support**: Add connectors for other popular LLMs (Claude, Mistral) and vector databases (FAISS, Pinecone, Weaviate).
-   [ ] **Advanced RAG Strategies**: Implement techniques like re-ranking (`Cross-Encoders`) or query transformations (`HyDE`) to improve retrieval accuracy.
-   [ ] **UI Enhancements**:
    -   Allow users to manage multiple chat sessions.
    -   Incorporate a user feedback mechanism (e.g., thumbs up/down on responses).
    -   Visualize the source documents and highlight the exact text chunks used.
-   [ ] **RAG Evaluation Pipeline**: Integrate a framework like RAGAs to quantitatively measure the performance of the retrieval and generation components.
-   [ ] **Containerization**: Provide a `Dockerfile` and `docker-compose.yml` for easy, reproducible deployment.

---
