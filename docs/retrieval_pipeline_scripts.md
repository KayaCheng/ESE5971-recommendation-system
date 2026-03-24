# Vector + Graph Pipeline Scripts

This repository now includes runnable scripts for:

- Vector storage build
- Concept extraction
- Neo4j graph ingestion
- Vector-graph binding
- Hybrid retrieval (vector recall + graph expansion)

## 0) Environment Setup

```bash
cd "/Users/thunder/Documents/ESE Practiculum/code/ESE5971-recommendation-system"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Build Vector Storage

### Local dry-run backend (`hash`)

```bash
python3 scripts/build_vector_store.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --output-dir storage/vector \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384
```

### Pretrained local embedding model (`sentence-transformers`)

```bash
python3 scripts/build_vector_store.py \
  --embedding-backend sentence_transformers \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --embedding-dim 384
```

### OpenAI embedding

```bash
export OPENAI_API_KEY="<your_key>"
python3 scripts/build_vector_store.py \
  --embedding-backend openai \
  --embedding-model text-embedding-3-small \
  --embedding-dim 1536
```

### Outputs

- `storage/vector/index.npy`
- `storage/vector/id_map.json`
- `storage/vector/metadata.sqlite`
- `storage/vector/manifest.json`

## 2) Extract Concepts / Relations

### Local heuristic backend

```bash
python3 scripts/extract_concepts.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --output-dir storage/graph \
  --backend heuristic
```

### OpenAI backend

```bash
export OPENAI_API_KEY="<your_key>"
python3 scripts/extract_concepts.py \
  --backend openai \
  --llm-model gpt-4.1-mini \
  --min-confidence 0.6
```

### Outputs

- `storage/graph/concepts.jsonl`
- `storage/graph/relations.jsonl`
- `storage/graph/extraction_log.jsonl`
- `storage/graph/manifest.json`

## 3) Start Neo4j (Local)

### Option A: Docker

```bash
docker run -d \
  --name mis-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j_password \
  neo4j:5
```

Set env vars:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4j_password"
export NEO4J_DATABASE="neo4j"
```

## 4) Ingest Graph into Neo4j

```bash
python3 scripts/ingest_graph_neo4j.py \
  --chunks-path data/processed/chunks/mis1to66_chunks.jsonl \
  --concepts-path storage/graph/concepts.jsonl \
  --relations-path storage/graph/relations.jsonl
```

This creates:

- `(:Chunk)` nodes
- `(:Concept)` nodes
- `(:Chunk)-[:MENTIONS]->(:Concept)`
- `(:Concept)-[:RELATES]->(:Concept)`
- `(:Chunk)-[:NEXT]->(:Chunk)`

## 5) Bind Vector and Graph Layers

```bash
python3 scripts/link_vector_graph.py \
  --vector-dir storage/vector \
  --graph-dir storage/graph \
  --index-name mis1to66_vector_index
```

This writes:

- SQLite tables in `storage/vector/metadata.sqlite`:
  - `vector_index_map`
  - `chunk_concept_map`
- Neo4j links:
  - `(:VectorIndex)` node
  - `(:Chunk)-[:IN_VECTOR_INDEX]->(:VectorIndex)`
  - and chunk vector properties (`vector_row_index`, `embedding_model`, `embedding_dim`)

## 6) Hybrid Retrieval

### Vector-only (no Neo4j expansion)

```bash
python3 scripts/hybrid_retrieve.py \
  --query "How does CT differ from MRI?" \
  --vector-dir storage/vector \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --top-k 5 \
  --no-graph
```

### Vector + Graph expansion

```bash
python3 scripts/hybrid_retrieve.py \
  --query "How does CT differ from MRI?" \
  --vector-dir storage/vector \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --top-k 5
```

Optional output file:

```bash
python3 scripts/hybrid_retrieve.py \
  --query "X-ray attenuation" \
  --no-graph \
  --output-json storage/retrieval/query_result.json
```

## Notes

- `hash` backend is for plumbing tests only, not semantic quality.
- For real semantic search use:
  - `sentence_transformers` for local pretrained models, or
  - `openai` for API embeddings.
- Hybrid retrieval requires embedding settings that match the built index.

## 7) Online Bandit Training (Replay -> Train -> Evaluate)

### Step A: Generate replay log with propensity + reward

```bash
python3 scripts/bandit_simulate_replay.py \
  --vector-dir storage/vector \
  --events-out storage/bandit/events.jsonl \
  --rounds 200 \
  --top-k 10 \
  --policy retrieval_epsilon \
  --epsilon 0.2
```

### Step B: Train LinUCB online from event stream

```bash
python3 scripts/bandit_train_online.py \
  --events-path storage/bandit/events.jsonl \
  --model-path storage/bandit/model_linucb.json \
  --reset
```

### Step C: Offline baseline comparison (IPS/SNIPS/DR)

```bash
python3 scripts/bandit_eval_offline.py \
  --events-path storage/bandit/events.jsonl \
  --model-path storage/bandit/model_linucb.json \
  --output-json storage/bandit/offline_eval_report.json
```

### Outputs

- `storage/bandit/events.jsonl`
- `storage/bandit/model_linucb.json`
- `storage/bandit/offline_eval_report.json`
