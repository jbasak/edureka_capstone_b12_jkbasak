# Procurement and Vendor Evaluation Assistant

This repository provides a grounded procurement assistant for RFP responses, pricing documents, and compliance workbooks. It uses FastAPI for the backend, Streamlit for the user interface, ChromaDB for vector search, `sentence-transformers/all-MiniLM-L6-v2` for embeddings, and Groq for answer generation.

## Architecture

1. Uploads are parsed from PDF, TXT, CSV, or XLSX into page/sheet-aware sections.
2. Sections are split into approximately 500-word chunks with a 50-word overlap and stored in ChromaDB.
3. The execution agent plans the question, retrieves relevant chunks, calls Groq with grounded context, and validates citations.
4. The compliance agent evaluates Vendor A, Vendor B, and Vendor C workbook columns as Pass, Fail, or Partially Met.

## Local setup without Docker

The application can run locally with Python 3.12, a local ChromaDB server, and three PowerShell terminals.

1. Create and activate a virtual environment:

	```powershell
	py -3.12 -m venv .venv
	.\.venv\Scripts\Activate.ps1
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	```

2. Create the environment file and update the local ChromaDB settings:

	```powershell
	Copy-Item .env.example .env
	```

	In `.env`, set these values for a local run and provide your Groq key:

	```dotenv
	GROQ_API_KEY=your_groq_api_key
	CHROMADB_HOST=localhost
	CHROMADB_PORT=8001
	API_BASE_URL=http://localhost:8000
	```

3. Start ChromaDB in terminal 1. The `--path` directory stores vectors locally and is created automatically:

	```powershell
	cd  C:\BASAK\Codebase\github_repos\edureka_capstone_b12_jkbasak
	.\.venv\Scripts\Activate.ps1
	chroma run --path .\chroma_data --host localhost --port 8001
	```

4. Start the FastAPI backend in terminal 2:

	```powershell
	cd  C:\BASAK\Codebase\github_repos\edureka_capstone_b12_jkbasak
	.\.venv\Scripts\Activate.ps1
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000
	```

5. Start the Streamlit UI in terminal 3:

	```powershell
	cd  C:\BASAK\Codebase\github_repos\edureka_capstone_b12_jkbasak
	.\.venv\Scripts\Activate.ps1
	streamlit run src/ui/app.py --server.port 8501
	```

Open the UI at `http://localhost:8501`. The API health check is available at `http://localhost:8000/health`, and ChromaDB listens on `http://localhost:8001`.

The first document upload downloads the `sentence-transformers/all-MiniLM-L6-v2` model, so embedding initialization may take a few minutes. Keep the ChromaDB, FastAPI, and Streamlit terminals running while using the application.

Full Docker deployment:

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY in .env
docker compose up --build
```

The UI is available at `http://localhost:8501`, the API at `http://localhost:8000`, and ChromaDB is persisted in the `chroma_data` volume.

## API

- `POST /documents`: multipart upload for PDF, TXT, CSV, or XLSX files.
- `POST /requirements`: multipart XLSX compliance evaluation.
- `POST /query`: grounded natural-language question with citations.
- `GET /health`: API liveness check.

## Tests and limitations

Run `pytest`. Retrieval and generation tests should mock ChromaDB and Groq; neither a live vector service nor an API key is required for unit tests. The chunker uses whitespace words as a lightweight token approximation, and generated answers remain decision support rather than legal or contractual advice