# 1. Overview
The goal of this capstone project is to develop a Generative AI–powered application that
enables users to query enterprise documents using autonomous AI agents. The system uses
Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Agentic AI
frameworks to retrieve relevant information, reason over it, and generate accurate, contextaware responses

# 2. Project Description
This project aims to build an AI agent–based knowledge and decision support system to work as Procurement & Vendor Evaluation Assistant. The application allows users to upload documents in multiple formats (PDF, TXT, CSV, Excel) and ask natural language questions. The system retrieves relevant content using a vector database and generates grounded responses using an LLM.

AI agents are used to plan the task, retrieve information, reason over the retrieved context,
and validate the final output, demonstrating a full-fledged Generative AI and Agentic AI
workflow

# 3. Project Specification:
## 3.1 Overview & ContextProblem Statement
Procurement and vendor management teams often spend hours or days manually scanning long, unstructured request-for-proposal (RFP) responses, complex pricing sheets, and master services agreements (MSAs) to ensure compliance and evaluate quotes.

## 3.2 Goals
* Automatically extract, normalize, and query compliance criteria from diverse vendor documents (PDF, TXT, CSV, XLSX).
* Leverage a multi-agent system to reason across requirements, cross-reference pricing models, and identify missing or non-compliant sections.
* Provide a grounded response backed by strict source citations to eliminate hallucinations.
* Package the application inside a reproducible Docker environment driven by a FastAPI backend.

## 3.3 Non-Goals
* This tool will not handle direct vendor communications or automated contract signing.
* It does not replace human legal review; it acts strictly as an analytical decision-support tool.

# 4. Functional Requirements
## 4.1 Document Handling & Chunking
* [ ] **Multi-Format Support:** Correctly parse text from raw `.txt`, `.pdf` layout files, `.csv` sheets, and multi-tab Excel files (`.xlsx`).
* [ ] **Overlapping Sliders:** Chunk sizes should target ~500 tokens with a 10% overlap to preserve bounding sentence contexts.

## 4.2 Agent Execution & Evaluation Guardrails
* [ ] **Requirement Checklist Extractor:** Given an Excel sheet of compliance requirements, map whether Vendor A, B, or C satisfies the condition (e.g., "Pass", "Fail", or "Partially Met").
* [ ] **Hallucination Minimization:** The system must return an explicit *"Information not found in context"* string if the Vector DB does not provide an explicit answer.
* [ ] **Strict Citation Injector:** Every analytical text output must include bracketed citations linked to the metadata payload (e.g., `[VendorA_Proposal.pdf, Page 12]`).


# 5. Technical Architecture & Stack
* **Language/Framework:** Python 3.12 + Stremlit based UI to provide users a Chat like interface.
* **Vector DB:** ChormaDB 
* **AI Model Routing:** OpenAI-compatible API configuration (configurable via local `.env` variables). Use GROQ API Key and Hugging face embeddig model entence-transformers/all-MiniLM-L6-v2.

* **Infrastructure:** Minimalistic `docker-compose.yml` defining the FastAPI worker service and the Qdrant instance sidecar.


# 6. Verification & Acceptance Criteria
These operational criteria must pass successfully during localized testing:
* [ ] **Scenario 1 (Ingestion):** Ppload payload sent to `/documents`, then the system successfully commits vectors to ChromaDB and yields a `201 Created` status.
* [ ] **Scenario 2 (Comparative Search):** Given a multi-vendor dataset, when queried with *"Compare annual pricing"*, then the system yields a markdown table outlining costs explicitly matching the input spreadsheet values.
* [ ] **Scenario 3 (Citation Validation):** When a response is generated, then every factual claim contains at least one target file source citation.


# 7. Tasks to be included as a part of the project
1. Set up the project foundation – Initialize the project repository, environment configuration, and basic application structure for the Generative AI system.
2. Design the user interaction layer – Create a simple interface or API that allows users to upload documents and ask natural language questions.
3. Implement document ingestion – Enable uploading and processing of enterprise documents in multiple formats such as PDF, TXT, CSV, or Excel.
4. Prepare data for semantic search – Convert processed document content into chunks suitable for embedding and retrieval.
5. Build a vector-based knowledge store – Generate embeddings and store them in a vector database to support semantic similarity search.
6. Implement intelligent document retrieval – Retrieve the most relevant document content based on user queries using similarity search.
7. Develop a Retrieval-Augmented Generation pipeline – Combine retrieved document context with an LLM to generate accurate, grounded responses.
8. Implement agent-based reasoning – Create one or more AI agents that plan, retrieve, reason, and generate responses using available tools.
9. Add reliability and safety controls – Handle errors, validate inputs, and apply guardrails to reduce hallucinations and unsafe outputs.
10. Deploy and document the solution – Deploy the application and provide documentation explaining the architecture, workflow, and limitations.

# 8. Submission Guidelines:
• Submit the complete source code and a documentation file in a Zip format.
• Documentation should explain system setup, architecture, agent roles, and deployment steps, along with limitations and challenges faced during development.