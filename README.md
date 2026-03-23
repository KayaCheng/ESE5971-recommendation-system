### GitHub repo link: https://github.com/KayaCheng/ESE5971-recommendation-system/tree/main

## 1. Introduction

### Decisions to be impacted
This project is designed to support content-selection and recommendation decisions in a medical imaging learning or knowledge-support setting. The system is intended to improve the following decisions:

- **What content should be recommended next?**
  Given a user query, learning topic, or clinical interest, the system should recommend the most relevant document chunks, references, or learning materials.
- **Which source is most relevant for a given imaging concept?**
  For example, when the topic is Fourier transform, morphological operators, CT principles, or MRI basics, the system should rank the most useful materials higher.
- **How should content be organized for retrieval and downstream recommendation?**
  The project needs a reliable pipeline for extracting, cleaning, chunking, embedding, and ranking educational resources.
- **Which modeling approach gives the best tradeoff between quality, simplicity, and explainability?**
  This includes deciding whether a simple lexical baseline, a classical machine learning model, or an embedding-based retrieval pipeline is the best starting point.

### Business value
Although this is a course project, the practical value is strong:

- **Faster access to relevant medical imaging knowledge.** Users spend less time searching through long PDFs and more time reading the most relevant sections.
- **Better educational support.** Students can discover the right chapter, concept explanation, or technical background more efficiently.
- **Foundation for clinical decision support extensions.** A strong retrieval and recommendation pipeline can later support more advanced tools such as case-based recommendations or multimodal search.
- **Reusable data pipeline.** The preprocessing pipeline can be extended from PDF documents to web pages, lecture notes, videos, and other medical imaging resources.

### Why we care about this project
We care about this project because medical imaging is a knowledge-dense domain: textbooks, notes, and references are often long, technical, and difficult to search manually. A recommendation system can reduce friction in learning and information discovery. It also gives us a realistic course project that combines data engineering, information retrieval, machine learning, and system design in one end-to-end workflow.

## 2. Data and Data Preprocessing

### Current data scope
The current repository is centered on PDF-based educational material, especially a source PDF located under `data/raw/pdf/`. The current processing pipeline converts the raw PDF into:

- **Page-level cleaned records** saved as JSONL
- **Chunk-level records** saved as JSONL for downstream retrieval, embedding, and recommendation

### Current preprocessing pipeline
The implemented PDF pipeline has three main stages:

1. **Extraction**
   - Open the PDF with PyMuPDF
   - Extract raw text page by page
   - Store page number, source name, and raw text

2. **Cleaning**
   - Normalize line breaks
   - Replace PDF ligatures such as `ﬁ`, `ﬂ`
   - Remove standalone page numbers
   - Merge broken hyphenation across lines
   - Merge wrapped lines into paragraph-like text
   - Keep the main body pages only
   - Compute cleaned text length (`char_count`)

3. **Chunking**
   - Split cleaned pages into paragraph-like units
   - Detect likely titles and headings
   - Skip extremely short low-information pages
   - Build chunk records with chunk metadata
   - Link neighboring chunks using `prev_chunk_id` and `next_chunk_id`

### Data cleaning metrics we can report
For the project report, we should report simple and interpretable cleaning metrics before and after preprocessing. Recommended metrics include:

- Number of raw pages extracted
- Number of body pages retained after filtering
- Average characters per page before and after cleaning
- Number of pages removed for being empty or low-information
- Average chunk length in characters
- Estimated average chunk length in tokens
- Number of total chunks produced
- Percentage of chunks that fall within the target size range

### Outlier detection ideas
Outlier detection is important because noisy pages can hurt retrieval quality. We recommend labeling the following as outlier candidates:

- Pages with extremely low character count
- Pages containing mostly numbers or formatting artifacts
- Pages with repeated headers/footers
- Pages with very high symbol ratio or OCR noise
- Chunks that are too short to be meaningful
- Chunks that are too long and likely mix multiple concepts

Recommended quantitative thresholds for a first pass:

- **Low-information page:** fewer than 80 characters
- **Preferred chunk range:** roughly 600 to 1800 characters
- **Flagged chunk outlier:** below 300 characters or above 2200 characters

### Metrics to judge whether chunking is good
Chunk quality should not be judged by chunk length alone. We should track a mix of **structural metrics**, **retrieval metrics**, and **manual review metrics**.


## 3. Problem Formulation

We frame this project as a **content recommendation and retrieval problem** for medical imaging knowledge sources.

### Input
One or more of the following:
- A user query
- A target topic or chapter
- A learning objective
- A clinical keyword or imaging concept

### Output
A ranked list of the most relevant document chunks, pages, or resources.

### Core machine learning question
Given a query or user need, **which chunk or document should be ranked highest?**

This means the project can be approached as:
- **Information retrieval**
- **Learning-to-rank**
- **Semantic search**
- **Recommendation over educational content**

---

## 4. Model Selection Roadmap

To keep the project disciplined, model selection should move from the simplest explainable baseline to stronger semantic models.

### Stage 1: Non-neural lexical baseline
**Goal:** establish the minimum viable retrieval benchmark.

Candidate methods:
- TF-IDF + cosine similarity
- BM25 ranking

Why start here:
- Easy to implement
- Fast to evaluate
- Strong interpretability
- Gives a clean benchmark for later models

What this baseline tells us:
- Whether exact keyword overlap is already enough for many queries
- Which query types fail without semantic understanding

### Stage 2: Classical supervised ranking / classification baseline
**Goal:** test whether handcrafted features or simple supervised models improve ranking.

Candidate methods:
- Logistic Regression
- Linear SVM
- XGBoost or LightGBM on engineered features

Possible features:
- TF-IDF similarity
- BM25 score
- Query-chunk token overlap
- Presence of title keywords
- Chunk length
- Section position in document

Why this stage matters:
- It helps us understand whether ranking gains come from better features rather than a more complex neural model.

### Stage 3: Embedding-based semantic retrieval
**Goal:** move beyond exact match and capture concept similarity.

Candidate methods:
- Sentence-BERT style embeddings
- Instructor-style embeddings
- Domain-adapted biomedical embeddings if available

How it works:
- Encode chunk text into vectors
- Encode the query into the same vector space
- Retrieve nearest chunks using cosine similarity or vector search

Why this stage matters:
- Medical imaging queries may use different words than the source text
- Semantic embeddings can retrieve useful content even when exact wording differs

### Stage 4: Re-ranking model
**Goal:** improve top-k ranking quality after initial retrieval.

Candidate methods:
- Cross-encoder re-ranker
- Small transformer-based relevance scorer
- Learning-to-rank model using retrieved candidates

Pipeline idea:
1. Use BM25 or embeddings to get top-k candidates
2. Use a stronger model to re-rank those candidates
3. Return the best final recommendations

### Stage 5: Full recommendation system extension
After retrieval quality is acceptable, the system can be extended to include:
- User profile personalization
- Topic progression recommendations
- Resource diversity constraints
- Multimodal recommendations from PDF, web, and video content

---

## 5. Baseline Model Ideas

Here are the baseline models I recommend, in order of importance.

### Baseline 1: BM25 retrieval baseline
**My strongest recommendation for the first baseline.**

Why BM25 should be the main baseline:
- It is a standard retrieval benchmark
- It works well on chunked text documents
- It is simple, explainable, and easy to debug
- It gives a very strong non-neural reference point
- If a more advanced model cannot beat BM25, then the added complexity is not justified

**Use case:**
- Input: a query such as “what is convolution in imaging?”
- Output: rank all chunks by BM25 score and return the top-k chunks

**What to evaluate:**
- Precision@k
- Recall@k
- Mean Reciprocal Rank (MRR)
- Qualitative relevance of top returned chunks



### Process
```bash
# 0) Project base
cd "../ESE Practiculum/code/ESE5971-recommendation-system"

# 1) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Update pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run “Without OpenAI” heuristic process
python3 scripts/build_vector_store.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --output-dir storage/vector \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384

python3 scripts/extract_concepts.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --output-dir storage/graph \
  --backend heuristic \
  --min-confidence 0.6

# 4) Check if output files exist
find storage -maxdepth 3 -type f | sort

# 5) Quick check output results
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("storage/vector/manifest.json")
print("vector manifest exists:", p.exists())
if p.exists():
    print(json.loads(p.read_text())["num_chunks"])

p = pathlib.Path("storage/graph/manifest.json")
print("graph manifest exists:", p.exists())
if p.exists():
    m = json.loads(p.read_text())
    print("chunks_processed:", m["chunks_processed"], "concepts_total:", m["concepts_total"], "relations_total:", m["relations_total"])
PY

# 6) （Optional）If OpenAI token exists then run the real model
export OPENAI_API_KEY="sk-xxxx"

python3 scripts/build_vector_store.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --output-dir storage/vector_openai \
  --embedding-backend openai \
  --embedding-model text-embedding-3-small \
  --embedding-dim 1536

python3 scripts/extract_concepts.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --output-dir storage/graph_openai \
  --backend openai \
  --llm-model gpt-4.1-mini \
  --min-confidence 0.6

# 7) Check OpenAI mode output
find storage/vector_openai storage/graph_openai -maxdepth 2 -type f | sort
```


## 10. Where to Start Next for a RAG Recommendation System

If the next phase of the project is to build a **RAG-based recommendation system**, we should start from the simplest end-to-end pipeline and then improve it step by step.

### Step 1: Freeze the chunking pipeline and create a stable corpus
Before building RAG, we need a stable document collection.

Immediate tasks:
- Run the PDF pipeline end to end
- Save the cleaned page-level JSONL
- Save the chunk-level JSONL
- Inspect chunk quality using the chunk metrics above
- Finalize a first chunking configuration before indexing

Why this is first:
- RAG quality depends directly on the quality of the chunks
- If chunk boundaries change every day, retrieval evaluation becomes unreliable

### Step 2: Build the first retriever
The retriever is the most important component in an early RAG system.

Recommended order:
1. **BM25 retriever** as the first baseline
2. **Embedding retriever** as the first semantic retriever
3. **Hybrid retriever** combining BM25 and embeddings

Suggested implementation idea:
- Input: user query
- Retrieve: top-k chunks from the indexed corpus
- Output: ranked chunk candidates for generation or recommendation

Why start here:
- In RAG, poor retrieval usually hurts the system more than weak generation
- A strong retriever gives us a usable system even before adding an LLM

### Step 3: Define what “recommendation” means in the RAG setting
For this project, recommendation should be defined explicitly. We have at least three possible recommendation targets:

#### Option A: Recommend relevant chunks
Given a query, return the most relevant text chunks.

Use when:
- the goal is search and knowledge support
- we want the simplest RAG prototype

#### Option B: Recommend source documents or chapters
Given a query, aggregate chunk scores and recommend the best pages, sections, or source documents.

Use when:
- the user wants a full chapter, source, or reading recommendation

#### Option C: Recommend learning paths or related topics
Given a topic, recommend what the user should read next.

Use when:
- the project wants a more educational recommender flavor
- we want topic progression rather than only retrieval

**My recommendation:** start with **Option A (chunk recommendation)** first, then extend to **Option B (document recommendation)** after retrieval works well.

### Step 4: Add the generator only after retrieval is working
Once retrieval is stable, then add the generation layer.

Basic RAG flow:
1. User submits a query
2. Retriever gets top-k chunks
3. Prompt builder combines the query and retrieved chunks
4. LLM generates:
   - an answer
   - a recommended reading list
   - or a justification for why those chunks were selected

What the generator should do in our project:
- Summarize the retrieved content
- Explain why a chunk or document is relevant
- Turn retrieval results into a more user-friendly recommendation response

### Step 5: Add recommendation-specific ranking logic
A RAG pipeline becomes a recommendation system when we go beyond plain retrieval and add ranking logic that reflects user needs.

Possible ranking signals:
- Retrieval score
- Source quality or chapter importance
- Topic coverage and diversity
- Difficulty level
- User history or interest profile
- Novelty versus redundancy

Example:
- Query: “I want to study convolution in imaging”
- Retrieval gets relevant chunks
- Recommendation layer reorders them into:
  1. introductory explanation
  2. mathematical definition
  3. applications in imaging
  4. follow-up topic: correlation vs convolution

This is where the system becomes more than search.

### Step 6: Build a small evaluation set before scaling
Before building a complex RAG system, create a small but high-quality evaluation benchmark.

Recommended evaluation units:
- **Retrieval evaluation**: are the right chunks retrieved?
- **Recommendation evaluation**: are the recommended chunks/documents actually useful?
- **Generation evaluation**: is the final answer grounded in retrieved evidence?

Minimum benchmark suggestion:
- 20 to 30 representative medical imaging queries
- manually labeled relevant chunks
- optional relevance grades: highly relevant / somewhat relevant / irrelevant

### Step 7: Start with a minimal system architecture
The first version does not need to be complicated.

Recommended MVP architecture:

1. **Ingestion layer**
   - PDF extraction
   - cleaning
   - chunking

2. **Indexing layer**
   - BM25 index
   - vector index for embeddings

3. **Retrieval layer**
   - top-k lexical retrieval
   - top-k semantic retrieval
   - optional hybrid score fusion

4. **Generation / recommendation layer**
   - prompt template
   - LLM response
   - recommendation explanation

5. **Evaluation layer**
   - query set
   - retrieval metrics
   - manual review

### Step 8: Concrete implementation order for the next 2 to 3 weeks

#### Week 1: Make retrieval work
- Finalize chunk output
- Build BM25 index over chunk text
- Build a query interface
- Return top-5 chunks for each query
- Create 10 to 20 manual test queries

#### Week 2: Add semantic retrieval
- Generate embeddings for each chunk
- Build a vector index
- Compare BM25 vs embedding retrieval
- Analyze failure cases

#### Week 3: Build the first RAG prototype
- Feed top-k retrieved chunks into an LLM prompt
- Generate answer + recommended reading chunks
- Add citation of chunk IDs or page numbers
- Evaluate grounding quality

### Step 9: My recommended starting point
If you ask me what to do **right now**, I would start with these three tasks:

1. **Create a fixed chunk dataset**
2. **Implement BM25 retrieval over chunks**
3. **Build a tiny labeled query set for evaluation**

Reason:
- this gives us the first working retrieval backbone
- it matches the current state of the repository
- it is the cleanest foundation for a later RAG system

### Step 10: Best first RAG deliverable
The best first deliverable is not a fancy chatbot. It is:

> **A system that takes a medical imaging query, retrieves the top relevant chunks, and returns a short evidence-grounded recommendation with source references.**

That deliverable is realistic, demonstrable, and easy to evaluate.

---

## 11. Current Status

At the current stage, the strongest implemented part of the project is the **PDF preprocessing pipeline**, which extracts text, cleans pages, and creates structured chunks. The next major milestone should be building and evaluating the first retrieval baseline, then turning that retriever into the backbone of a **RAG-based recommendation system**, with **BM25 as the official first baseline** and **hybrid retrieval as the strongest practical early-system direction**.

---

## 12. Updated System Roadmap: Vector Store + Knowledge Graph + Bandit Recommendation

Our updated direction is no longer just “retrieve the most relevant chunks.”  
The new goal is to build a **dual-memory recommendation system**:

- a **vector database** for semantic retrieval of chunk content
- a **knowledge graph** for concept structure and learning dependencies
- a **bandit-based recommendation layer** for selecting the best next concept or content for a user

In this setup, the JSONL chunk output is the starting point, not the final product.

### Stage 1: Vector Storage for Semantic Search

The first task is to turn each chunk into an embedding and store it in a vector database.

#### Input
- chunk-level JSONL produced by the preprocessing pipeline
- each record should include at least:
  - `chunk_id`
  - `source_name`
  - `chunk_text`

#### Process
1. Read the JSONL file line by line
2. Feed `chunk_text` into an embedding model
3. Store the embedding in a vector database such as **Qdrant** or **Milvus**
4. Save metadata together with the vector

#### Required metadata / payload
When inserting vectors, the payload should include:
- `chunk_id`
- `source_name`
- `page_start`
- `page_end`
- optional future fields such as:
  - `concepts`
  - `difficulty`
  - `section_title`

#### Recommended embedding model
For the first implementation, a strong default choice is:
- **BGE-m3**

Why:
- good multilingual and retrieval performance
- suitable for semantic search
- strong practical baseline for chunk-level embedding

#### Why this stage matters
This stage solves the **semantic retrieval** problem:
- users will be able to search by meaning, not only by exact keyword match
- retrieved chunks can later be used as recommendation candidates or LLM context

### Stage 2: Concept Extraction for Knowledge-Graph Nodes

The second task is to “dehydrate” each chunk into a smaller set of educational concepts.

#### Goal
Transform chunk text into structured knowledge units such as:
- concepts
- algorithms
- modalities
- prerequisites
- difficulty level

#### Process
1. Send each chunk to an LLM
2. Use a fixed output schema
3. Extract concept-level entities from the chunk
4. Store the extracted concepts together with the originating `chunk_id`

#### Example extraction target
If a chunk explains Fourier Transform in CT, the model could output structured information such as:

```json
{
  "concept": "Fourier Transform",
  "algorithm": "Filtered Back Projection",
  "difficulty": 4,
  "mentioned_in_chunk": "mis1to66_chunk_0012"
}
```

#### Recommended implementation choice
- LLM for extraction: **GPT-4o**
- Output format: strict JSON
- Extraction style: schema-guided prompting

#### Why this stage matters
This stage prepares the **nodes** of the graph database.  
Instead of storing only raw text, we begin storing explicit knowledge units that can support sequencing, prerequisite reasoning, and personalized recommendation.

### Stage 3: Relationship Discovery for Graph Edges

The third task is to discover logical relations between concepts.

#### Goal
Build the “edges” of the knowledge graph, especially educational relationships such as:
- `PREREQUISITE_OF`
- `PART_OF`
- `USED_IN`
- `EXTENDS`
- `RELATED_TO`

#### Example
If one chunk explains that Radon Transform is a foundation for CT reconstruction, we should create:

```text
(Radon Transform) -[:PREREQUISITE_OF]-> (CT Reconstruction)
```

If a chunk explains that Filtered Back Projection is part of Image Reconstruction, we should create:

```text
(Filtered Back Projection) -[:PART_OF]-> (Image Reconstruction)
```

#### How to discover relationships
Two practical options:

1. **LLM-based extraction**
   - ask the model to infer concept relationships from chunk text
   - best for nuanced educational logic

2. **Heuristic / rule-based extraction**
   - use phrase patterns such as:
     - “is the basis of”
     - “is part of”
     - “is used in”
   - useful for bootstrapping and validation

#### Why this stage matters
This stage gives us the **learning structure** of the domain:
- what should be learned first
- what belongs to a larger topic
- which concepts are adjacent in a learning path

### Stage 4: The Bridge Between Vector Store and Graph Database

This is the most important system-level connection in the new architecture.

We do not want the vector store and graph store to exist as separate silos.  
They must be linked in both directions.

#### In the graph database
Each concept node should contain a field such as:

- `mentioned_in_chunks: [chunk_id_1, chunk_id_2, ...]`

This tells us which text chunks support the concept.

#### In the vector database
Each chunk payload should store concept-level metadata such as:

- `concepts`
- `difficulty`
- `chapter`

This tells us which concepts are discussed in that chunk.

#### Recommended graph database
- **Neo4j**

#### Why this bridge matters
This bridge enables two-way reasoning:

- **From graph to content**
  - find a concept in the graph
  - get its supporting chunk IDs
  - retrieve the actual text from the vector store

- **From content to graph**
  - retrieve semantically relevant chunks
  - inspect which concepts they mention
  - reason over concept-level structure and prerequisites

This is the core integration that makes the system more than a normal RAG pipeline.

### Stage 5: Final Post-Processing State

After these steps, the system should have two aligned knowledge layers.

#### Vector database
Contains:
- embeddings for every chunk
- metadata such as chunk ID, source, page range, and concepts
- support for semantic retrieval

#### Graph database
Contains:
- concept nodes
- algorithm nodes
- difficulty or topic metadata
- prerequisite and part-of relations
- references back to chunk IDs

#### Intuition
At that point, the project will look like this:

- the vector database stores the **meaning of each chunk**
- the graph database stores the **logic of the domain**

For example, the graph may encode a sequence like:

```text
Signal Processing -> Fourier Transform -> Projection Theorem -> CT Reconstruction
```

This is exactly what we need for educational recommendation and adaptive sequencing.

### Stage 6: How This Supports Online Bandit Training

This new architecture is especially useful for online recommendation models such as **LinUCB**.

#### State
The system state can be built from:
- concepts the user has already studied
- graph neighbors of those concepts
- user difficulty level
- recent interaction history

#### Action
An action can be:
- recommending the next concept
- recommending the next chunk
- recommending the next reading path

A natural action set can be generated from:
- neighboring graph nodes
- prerequisite graph edges
- semantically similar chunks in the vector database

#### Content retrieval
Once the bandit chooses the next concept or concept-neighbor:
1. look up the concept in the graph
2. get the associated `chunk_id` list
3. retrieve the relevant chunk content from the vector database
4. return the content to the user

#### Why this matters
This means the bandit is not choosing from random content.  
It is choosing from a structured candidate set grounded in:
- concept dependencies
- semantic relevance
- user learning state

### Stage 7: Recommended Immediate Implementation Order

To make this roadmap actionable, I recommend the following order:

#### Phase A: Finish vector storage first
- read chunk JSONL
- embed `chunk_text`
- store vectors in Qdrant or Milvus
- save metadata payload for each chunk

#### Phase B: Extract concepts
- design the concept extraction schema
- run GPT-4o over chunks
- save concept JSON outputs
- keep `chunk_id` as the provenance key

#### Phase C: Build the graph
- create Neo4j nodes for concepts
- create edges such as `PREREQUISITE_OF` and `PART_OF`
- attach `mentioned_in_chunks`

#### Phase D: Build the bridge
- write concept names into vector-store metadata
- write chunk references into graph nodes

#### Phase E: Add recommendation logic
- use graph neighbors as candidate next concepts
- use the vector store to retrieve actual chunk content
- optionally add LinUCB or another contextual bandit for online learning

### Stage 8: Best First Deliverable Under the New Plan

The best first deliverable is:

> **A prototype that stores chunk embeddings in a vector database, extracts concepts into Neo4j, links concepts back to chunk IDs, and returns concept-grounded recommended reading chunks for a user query.**

This deliverable is strong because it already demonstrates:
- semantic retrieval
- structured knowledge extraction
- graph-based learning dependencies
- a clear path to personalized bandit recommendation
