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

`scripts/hybrid_retrieve.py` now supports two modes:

- `--mode annotate`:
  - old behavior
  - vector top-k retrieval + optional Neo4j context attachment
  - does **not** rerank by graph score
- `--mode rerank_lightgraph`:
  - local lightweight graph expansion + rerank
  - no Neo4j required
  - uses `metadata.sqlite` (`chunk_concept_map`) + `storage/graph/relations.jsonl`

### Mode A: annotate (vector-only, no graph)

```bash
python3 scripts/hybrid_retrieve.py \
  --query "How does CT differ from MRI?" \
  --mode annotate \
  --vector-dir storage/vector \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --top-k 5 \
  --no-graph
```

### Mode A: annotate (vector + Neo4j context)

```bash
python3 scripts/hybrid_retrieve.py \
  --query "How does CT differ from MRI?" \
  --mode annotate \
  --vector-dir storage/vector \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --top-k 5 \
  --neo4j-password "$NEO4J_PASSWORD"
```

### Mode B: rerank_lightgraph (vector -> graph expansion -> rerank)

```bash
python3 scripts/hybrid_retrieve.py \
  --query "How does CT differ from MRI?" \
  --mode rerank_lightgraph \
  --vector-dir storage/vector \
  --graph-dir storage/graph \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --top-k 5 \
  --seed-top-k 20 \
  --alpha 0.7 \
  --beta 0.3 \
  --output-json storage/retrieval/query_result_lightgraph.json
```

## Notes

- `hash` backend is for plumbing tests only, not semantic quality.
- For real semantic search use:
  - `sentence_transformers` for local pretrained models, or
  - `openai` for API embeddings.
- Hybrid retrieval requires embedding settings that match the built index.

## 6.5) Quick Lite-Graph Validation (No Docker / No Neo4j)

If you are short on time and only want to quickly test whether **graph+vector** can beat **vector-only**, use this local A/B path.

This path does **not** require Neo4j. It uses:

- `storage/vector/index.npy` (vector recall)
- `storage/vector/metadata.sqlite` table `chunk_concept_map` (chunk-concept mapping)
- `storage/graph/relations.jsonl` (concept relation propagation)

### Step A: Ensure local vector/graph mapping exists (skip Neo4j)

```bash
python3 scripts/link_vector_graph.py \
  --vector-dir storage/vector \
  --graph-dir storage/graph \
  --index-name mis1to66_vector_index \
  --skip-neo4j
```

### Step B: Run local A/B test (vector-only vs vector+light-graph)

```bash
python3 scripts/lightgraph_ab_test.py \
  --vector-dir storage/vector \
  --graph-dir storage/graph \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --top-k 5 \
  --seed-top-k 20 \
  --output-json storage/retrieval/lightgraph_ab_report.json
```

If you have real relevance labels (`qrels`), add:

```bash
python3 scripts/lightgraph_ab_test.py \
  --qrels-json data/processed/retrieval_qrels.json \
  --output-json storage/retrieval/lightgraph_ab_report.json
```

`qrels` format example:

```json
[
  {
    "query": "How does CT differ from MRI?",
    "relevant_chunk_ids": ["mis_001_c003", "mis_001_c019"]
  }
]
```

### Output

- `storage/retrieval/lightgraph_ab_report.json`
- Aggregate metrics for both methods:
  - `MRR`
  - `Recall`
  - `nDCG`
  - `delta_hybrid_minus_vector`

When `--qrels-json` is not provided, the script runs in `proxy_concept_match` mode (weak supervision). Treat this as a **fast directional check**, not final evidence.

## 6.7) Concept Learning Path (Before Online Bandit)

Use this step when you want to output a **concept sequence** instead of only chunk ranking.

Pipeline used by the script:

- query -> vector top-k chunk seeds
- seed chunks -> seed concepts
- concept graph expansion (from `relations.jsonl`)
- dependency-aware ordering (`depends_on` / `part_of` / `is_a`)
- final concept path + support chunks

```bash
python3 scripts/generate_concept_path.py \
  --query "How does CT differ from MRI?" \
  --vector-dir storage/vector \
  --graph-dir storage/graph \
  --embedding-backend hash \
  --embedding-model hash-v1 \
  --embedding-dim 384 \
  --seed-top-k 20 \
  --max-concepts 8 \
  --output-json storage/retrieval/concept_path_ct_mri.json
```

Output:

- `storage/retrieval/concept_path_ct_mri.json`
- fields:
  - `concept_path`: ordered learning concepts
  - `seed_chunks`: supporting retrieval evidence

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

## 8) MOOCCubeX Topic Filter + Contextual Events

Use this when you only want medical-imaging-related subset instead of full dataset.

### Step A: Filter MOOCCubeX to medical imaging subset

```bash
python3 scripts/filter_mooccubex_medical_imaging.py \
  --mooc-root /path/to/MOOCCubeX \
  --output-dir storage/mooccubex_medimg \
  --verbose
```

Optional custom keywords:

```bash
python3 scripts/filter_mooccubex_medical_imaging.py \
  --mooc-root /path/to/MOOCCubeX \
  --output-dir storage/mooccubex_medimg \
  --keywords "medical imaging" "radiology" "mri" "ct" "x-ray"
```

### Step B: Build contextual bandit events from filtered subset

```bash
python3 scripts/build_bandit_events_from_mooccubex.py \
  --filtered-dir storage/mooccubex_medimg \
  --events-out storage/mooccubex_medimg/bandit_events.jsonl \
  --top-k 10 \
  --min-user-events 3
```

### Step C: Train/evaluate on this subset

```bash
python3 scripts/bandit_train_online.py \
  --events-path storage/mooccubex_medimg/bandit_events.jsonl \
  --model-path storage/mooccubex_medimg/model_linucb.json \
  --reset

python3 scripts/bandit_eval_offline.py \
  --events-path storage/mooccubex_medimg/bandit_events.jsonl \
  --model-path storage/mooccubex_medimg/model_linucb.json \
  --output-json storage/mooccubex_medimg/offline_eval_report.json
```

### Outputs

- `storage/mooccubex_medimg/concepts_filtered.jsonl`
- `storage/mooccubex_medimg/videos_filtered.jsonl`
- `storage/mooccubex_medimg/courses_filtered.jsonl`
- `storage/mooccubex_medimg/problems_filtered.jsonl`
- `storage/mooccubex_medimg/events_filtered.jsonl`
- `storage/mooccubex_medimg/bandit_events.jsonl`
