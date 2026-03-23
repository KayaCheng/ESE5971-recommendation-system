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

### Baseline 2: TF-IDF + cosine similarity
This is the simplest baseline and should be used as a sanity check.

Why include it:
- Very easy to build
- Useful for debugging preprocessing and tokenization
- Helps compare vector-space lexical matching vs probabilistic lexical ranking (BM25)

Expectation:
- Likely weaker than BM25, but still useful as a very transparent baseline

### Baseline 3: Embedding retrieval baseline
This should be the **first semantic baseline** after lexical methods.

Recommended setup:
- Chunk the documents
- Build embeddings for each chunk
- Use cosine similarity to retrieve top-k candidates

Why it matters:
- It will show whether semantic similarity adds value beyond keyword matching
- It becomes the bridge to a more advanced recommendation pipeline

### Baseline 4: Hybrid baseline
A very practical project baseline is a **hybrid retriever**:
- BM25 score
- Embedding similarity score
- Weighted combination of both



ESE5971-recommendation-system/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── pdf/
│   │   ├── video/
│   │   └── web/
│   └── processed/
│       ├── cleaned/
│       └── chunks/
├── src/
│   ├── pdf_processing/
│   ├── video_processing/
│   ├── web_processing/
│   ├── chunking/
│   └── utils/
├── notebooks/
└── tests/

Phase 2:
Stage A: Vector Storage (Embedding + local index)
 - input:
   - chunks_jsonl: data/processed/chunks/mis1to66_chunks.jsonl
 - config:
   - embedding_model_name
   - embedding_dim
   - batch_size
 - output:
   - storage/vector/index.faiss
   - storage/vector/metadata.sqlite
   - storage/vector/id_map.json

Stage B: Concept Extraction (LLM schema extraction + preparation for knowledge graph)
- input:
   - metadata.sqlite (or chunks_jsonl)
 config:
   llm_model_name
   extraction_schema
 output:
   storage/graph/concepts.jsonl
   storage/graph/relations.jsonl
   storage/graph/extraction_log.jsonl


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
