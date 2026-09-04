Act as a Principal AI Engineer and Full-Stack Python Developer. Your task is to build a production-ready repository for a Generative AI Procurement & Vendor Evaluation Assistant based on the provided technical specification.

### Technical Stack Requirements:
- Language & Runtime: Python 3.12
- Web Framework: FastAPI (Backend Engine)
- Frontend UI: Streamlit (Interactive Chat & Upload Interface)
- Vector DB: ChromaDB
- Embeddings Model: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- LLM Provider: Groq API (OpenAI-compatible) configured via `.env`
- Containerization: `docker-compose.yml` defining the FastAPI application and persistent ChromaDB vector store sidecar

---

### Project Structure & File Layout:
Please generate the complete file structure and implementation code for the following modules:

1. `docker-compose.yml` & `Dockerfile`:
   - Setup services for the FastAPI app and ChromaDB.
2. `.env.example`:
   - Variables for `GROQ_API_KEY`, `CHROMADB_HOST`, `CHROMADB_PORT`, and `EMBEDDING_MODEL_NAME`.
3. `requirements.txt`:
   - All necessary dependencies (`fastapi`, `uvicorn`, `streamlit`, `chromadb`, `sentence-transformers`, `groq`, `pypdf`, `pandas`, `openpyxl`, `python-dotenv`, `pydantic`).
4. `src/ingestion/`:
   - `parsers.py`: Utility functions to extract text from `.pdf`, `.txt`, `.csv`, and multi-tab `.xlsx` files.
   - `chunker.py`: Document chunking logic targeting ~500 tokens per chunk with a 10% (~50 tokens) sliding window overlap, preserving metadata (filename, page/sheet number).
5. `src/vectorstore/`:
   - `chroma_client.py`: Initialization of ChromaDB client, collection management, and embedding wrapper using `sentence-transformers/all-MiniLM-L6-v2`.
6. `src/agents/`:
   - `retrieval_agent.py`: Handles vector search against ChromaDB.
   - `requirement_agent.py`: Parses compliance spreadsheets and generates matrix evaluations comparing Vendors A, B, and C with status tags ("Pass", "Fail", "Partially Met").
   - `guardrails.py`: Enforces safety logic:
     - Returns `"Information not found in context"` if vector relevance scores are below threshold.
     - Formats strict bracketed metadata citations (e.g., `[VendorA_Proposal.pdf, Page 12]`) on every output claim.
7. `src/api/main.py`:
   - FastAPI application containing:
     - `POST /documents`: Upload endpoint that parses, chunks, embeds, and stores documents, returning `201 Created` on success.
     - `POST /query`: Agent execution endpoint that takes user prompts, runs retrieval/reasoning agents, and returns grounded responses with citations.
8. `src/ui/app.py`:
   - Streamlit interface featuring a side panel for document uploads (PDF, TXT, CSV, XLSX) and a main chat interface communicating with the FastAPI endpoints.

---

### Implementation Instructions:
1. Ensure clean, modular, and type-hinted Python 3.12 code throughout all files.
2. Add explicit error handling for invalid file formats and missing environment variables.
3. Include inline code comments explaining agent execution flows and guardrails.
4. Keep functions self-contained and ready to execute. Start by outputting the file structure followed by the full code for each file.