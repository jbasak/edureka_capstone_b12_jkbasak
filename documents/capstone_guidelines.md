# 1. Project Objective
Build an AI agent-based knowledge and decision support system that:
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

# 2. Project Process Flow
Followin are list of Key activities involved:
    Documents 
    Ingestion -> Parsing -> Normalization -> Chunking
    Embeddings
    Reasoning Agent -> Grounded Answer
    Validation Agent
    Final Response


## 3. Idea- Procurement / Vendor Evaluation Assistant

**Problem:** 
Procurement teams need to compare vendors using contracts, proposals, pricing sheets, and requirements.

**Documents:**
- Vendor proposals PDF
- Pricing CSV
- Requirements Excel
- Contract TXT/PDF

**Example questions:**
- "Which vendor satisfies all mandatory requirements?"
- "Compare annual pricing."
- "Which contract contains an auto-renewal clause?"
- "Identify missing information."

**Agent roles:**
- Requirement Extraction Agent
- Retrieval Agent
- Comparison/Reasoning Agent
- Validation Agent


# 4. Reference Architecture

                         +----------------------+
                         |      User / UI       |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |      FastAPI API     |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Agent Controller   |
                         +----------+-----------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
              v                     v                     v
       +-------------+       +-------------+       +-------------+
       | Planner     |       | Retrieval   |       | Validator   |
       | Agent       |       | Tool/Agent  |       | Agent       |
       +------+------+       +------+------+       +------+------+
              |                     |                     |
              |                     v                     |
              |              +-------------+              |
              |              | Vector DB   |              |
              |              +------+------+              |
              |                     |                     |
              |                     v                     |
              |              Retrieved Chunks             |
              |                     |                     |
              +---------------------+---------------------+
                                    |
                                    v
                         +----------------------+
                         |   Reasoning / LLM    |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Grounded Response    |
                         | + Source Citations   |
                         +----------------------+