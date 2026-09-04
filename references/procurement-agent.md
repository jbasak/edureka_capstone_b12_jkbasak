# System Specification: Generative AI Procurement & Vendor Evaluation Assistant

---

## 1. Executive Summary & Context
* **Problem Statement:** Procurement and vendor management teams spend extensive time manually scanning unstructured RFP responses, pricing sheets, and Master Services Agreements (MSAs) to check compliance and evaluate quotes[cite: 1].
* **Solution Goal:** Build an agent-based decision-support system to upload multi-format documents, cross-reference vendor capabilities, analyze compliance, and retrieve grounded answers with source citations[cite: 1].
* **Scope:** 
  * **In-Scope:** Multi-format document parsing, multi-agent reasoning, compliance matrix extraction, vector search retrieval, citation injection, and reproducible Docker deployment[cite: 1].
  * **Out-of-Scope:** Direct vendor communication, automated contract signing, and replacing human legal review[cite: 1].

---

## 2. Technical Architecture & Tech Stack
+-----------------------------------------------------------------------------------+
|                                  Streamlit UI                                     |
+-----------------------------------------------------------------------------------+
|  (HTTP/REST)
v
+-----------------------------------------------------------------------------------+
|                                FastAPI Backend                                    |
|  +-----------------------------------------------------------------------------+  |
|  |                             Agent Execution Engine                          |  |
|  |   +-----------------------+   +-------------------+   +-----------------+   |  |
|  |   | Planning Agent        |   | Requirement Agent |   | Retrieval Agent |   |  |
|  |   +-----------------------+   +-------------------+   +-----------------+   |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|  +-------------------------------------+---------------------------------------+  |
|  |  Embeddings (all-MiniLM-L6-v2)     |  LLM Engine (Groq API Key / OpenAI)  |  |
|  +-------------------------------------+---------------------------------------+  |
+-----------------------------------------------------------------------------------+
|
v
+-----------------------------------------------------------------------------------+
|                         Vector Database (ChromaDB Container)                       |
+-----------------------------------------------------------------------------------+

* **Frontend UI:** Python 3.12 + Streamlit (Chat-based interface)[cite: 1].
* **Backend API:** FastAPI application service[cite: 1].
* **LLM Engine:** Groq API (OpenAI-compatible) via `.env` configuration[cite: 1].
* **Embeddings Model:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`[cite: 1].
* **Vector Store:** ChromaDB instance[cite: 1].
* **Deployment/Infra:** `docker-compose.yml` encapsulating the FastAPI application worker and vector store sidecar[cite: 1].

---

## 3. Functional Requirements

### Document Processing Pipeline
* **Supported Extensions:** Raw `.txt`, layout-preserved `.pdf`, `.csv` spreadsheets, and multi-tab `.xlsx` files[cite: 1].
* **Chunking Strategy:** ~500 tokens per chunk with a 10% overlapping slider (~50 tokens) to preserve contextual boundaries[cite: 1].

### Agent Workflow & Guardrails
* **Requirement Extraction Agent:** Evaluates compliance Excel sheets to extract criteria matrix mapping for Vendors A, B, and C (categorized as "Pass", "Fail", or "Partially Met")[cite: 1].
* **Hallucination Prevention Guardrail:** Returns explicitly `"Information not found in context"` if relevant vectors do not contain the necessary information[cite: 1].
* **Source Citation Injector:** Appends metadata-linked bracketed citations (e.g., `[VendorA_Proposal.pdf, Page 12]`) to every generated claim[cite: 1].

---

## 4. System Verification & Acceptance Criteria

| Scenario | Trigger / Action | Expected Result |
| :--- | :--- | :--- |
| **1. Ingestion** | Send document upload payload to `/documents` endpoint[cite: 1]. | System embeds/stores vectors in ChromaDB and returns HTTP `201 Created` status[cite: 1]. |
| **2. Comparative Search** | Query *"Compare annual pricing"* against multi-vendor files[cite: 1]. | Yields a structured Markdown table comparing pricing matching spreadsheet values[cite: 1]. |
| **3. Citation Check** | Query factual content across indexed documents[cite: 1]. | Every output statement contains bracketed target file citations[cite: 1]. |

---

## 5. Implementation Roadmap

1. **Foundation & Config:** Initialize repository structure, set up `.env` configurations (Groq API, ChromaDB settings), and establish Python 3.12 dependencies[cite: 1].
2. **User Interface:** Implement the Streamlit chat interaction layer supporting file upload components and dialogue history[cite: 1].
3. **Ingestion & Parsing:** Build multi-format parsers for PDF, TXT, CSV, and XLSX formats[cite: 1].
4. **Chunking & Vector Store:** Integrate `all-MiniLM-L6-v2` embedding generation with 500-token/10% overlap chunking and load vectors into ChromaDB[cite: 1].
5. **RAG Pipeline & Retrieval:** Set up similarity retrieval against ChromaDB queries[cite: 1].
6. **Agent & Guardrail Logic:** Develop autonomous agents for planning, requirement mapping, hallucination fallbacks, and citation formatting[cite: 1].
7. **FastAPI & Docker Deployment:** Wrap backend endpoints in FastAPI and construct `docker-compose.yml` for multi-container coordination[cite: 1].
8. **Final Packaging:** Compile full source code and complete architectural documentation into submission zip format[cite: 1].