# edureka_capstone_b12_jkbasak
Capstone Project - Batch B12 - Jayanta Basak

# Prompt to genarate codebase 
You are a data enginner working to build an python based AI agent-based knowledge and decision support system that helps Procurement teams need to compare vendors using contracts, proposals, pricing sheets, and requirements.

The Procurement team will use frontend application to upload following documents
- Vendor proposals PDF
- Pricing CSV
- Requirements Excel
- Contract TXT/PDF

Procurement team will use a chatbot type interface to get answer to various queries like
- "Which vendor satisfies all mandatory requirements?"
- "Compare annual pricing."
- "Which contract contains an auto-renewal clause?"
- "Identify missing information."


Following AI agents need to be created to automate 
- Requirement Extraction Agent
- Retrieval Agent
- Comparison/Reasoning Agent
- Validation Agent

The application should use GROQ API key for reasoning and hugging face transformer model 

FOllowing are key functionalities required 
1. Accepts documents in PDF, TXT, CSV, and Excel formats.
2. Extracts and normalizes their content.
3. Splits content into retrieval-friendly chunks.
4. Creates embeddings and stores them in a vector database.
5. Retrieves relevant evidence for natural-language questions.
6. Uses an LLM to produce grounded answers with citations or source references.
7. Uses one or more AI agents to plan the task, select tools, retrieve information, reason over evidence, and validate the answer.
8. Applies safety, validation, error handling, and hallucination-reduction controls.
9. Provides an API or simple web UI.
10. Runs reproducibly in Docker.
11. Includes automated tests and project documentation.


## Download documents from
https://www.michigan.gov/dtmb/procurement/contractconnect/bid-proposals

# Install python libraries
create pyproject.toml
pip install -e .

# Run from bash shell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload


cp .env.example .env

# configure LLM_MODEL, LLM_API_KEY, EMBEDDING_MODEL, EMBEDDING_API_KEY
docker compose up --build
API: http://localhost:8000/docs

# Run from bash shell to Upload:
curl -X POST http://localhost:8000/documents -F "file=@data/sample/acme_proposal.txt"

# Run Test from bash shell
curl -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"question":"Which vendor satisfies all mandatory requirements?"}'

