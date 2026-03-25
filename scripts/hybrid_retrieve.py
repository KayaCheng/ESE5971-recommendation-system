#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.common import read_jsonl


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def minmax_norm(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    if hi <= lo:
        return {k: 0.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


class HashEmbedder:
    def __init__(self, dim: int):
        self.dim = dim

    def _vector_from_text(self, text: str) -> np.ndarray:
        values: List[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = __import__("hashlib").sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                raw = int.from_bytes(digest[i : i + 4], "big", signed=False)
                values.append((raw / 2**32) * 2.0 - 1.0)
                if len(values) == self.dim:
                    break
            counter += 1
        return l2_normalize(np.asarray(values, dtype=np.float32))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector_from_text(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed.") from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, model_name: str, api_key: str | None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package is not installed.") from exc
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model_name, input=list(texts))
        vectors = [item.embedding for item in response.data]
        arr = np.asarray(vectors, dtype=np.float32)
        return np.vstack([l2_normalize(v) for v in arr])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval with 2 modes: annotate | rerank_lightgraph.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--vector-dir", type=Path, default=Path("storage/vector"))
    parser.add_argument("--graph-dir", type=Path, default=Path("storage/graph"))
    parser.add_argument("--metadata-db", type=Path, default=None, help="Defaults to <vector-dir>/metadata.sqlite")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["annotate", "rerank_lightgraph"], default="annotate")

    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "sentence_transformers", "openai"],
        default="hash",
    )
    parser.add_argument("--embedding-model", default="hash-v1")
    parser.add_argument("--embedding-dim", type=int, default=384)

    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--no-graph", action="store_true", help="Skip graph stage and return vector-only results.")
    parser.add_argument("--max-graph-edges", type=int, default=8)

    parser.add_argument("--seed-top-k", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for vector score in rerank_lightgraph.")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for graph score in rerank_lightgraph.")
    parser.add_argument("--relation-decay", type=float, default=0.35)
    parser.add_argument("--concept-boost-limit", type=int, default=16)
    parser.add_argument("--max-chunks-per-concept", type=int, default=40)

    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def get_embedder(args: argparse.Namespace):
    if args.embedding_backend == "hash":
        return HashEmbedder(dim=args.embedding_dim)
    if args.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=args.embedding_model)
    if args.embedding_backend == "openai":
        return OpenAIEmbedder(model_name=args.embedding_model, api_key=os.getenv("OPENAI_API_KEY"))
    raise ValueError(f"Unsupported embedding backend: {args.embedding_backend}")


def load_vector_artifacts(vector_dir: Path) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    index_path = vector_dir / "index.npy"
    id_map_path = vector_dir / "id_map.json"
    manifest_path = vector_dir / "manifest.json"

    for p in (index_path, id_map_path, manifest_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing vector artifact: {p}")

    matrix = np.load(index_path)
    with id_map_path.open("r", encoding="utf-8") as f:
        id_map = json.load(f)
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if matrix.shape[0] != len(id_map):
        raise RuntimeError("index.npy row count does not match id_map length.")

    return matrix.astype(np.float32), [str(x) for x in id_map], manifest


def vector_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.asarray([], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix_n = matrix / norms
    q = l2_normalize(query_vec.astype(np.float32))
    return matrix_n @ q


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def fetch_chunk_metadata(db_path: Path, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join(["?"] * len(chunk_ids))
        rows = conn.execute(
            f"""
            SELECT chunk_id, source_name, page_start, page_end, char_count, token_estimate, chunk_text
            FROM chunk_embeddings
            WHERE chunk_id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[row[0]] = {
            "source_name": row[1],
            "page_start": row[2],
            "page_end": row[3],
            "char_count": row[4],
            "token_estimate": row[5],
            "chunk_preview": (row[6] or "")[:350],
        }
    return out


def fetch_graph_context_for_chunk(session, chunk_id: str, max_edges: int) -> Dict[str, Any]:
    mentions_row = session.run(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(m:Concept)
        RETURN collect(DISTINCT {
            concept_id: m.concept_id,
            name: m.name,
            type: m.type
        }) AS mentions
        """,
        {"chunk_id": chunk_id},
    ).single()
    mentions = mentions_row["mentions"] if mentions_row and mentions_row["mentions"] else []

    related_row = session.run(
        """
        MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(m:Concept)
        MATCH (m)-[r:RELATES]->(n:Concept)
        RETURN collect(DISTINCT {
            source_concept_id: m.concept_id,
            source_name: m.name,
            relation: r.type,
            target_concept_id: n.concept_id,
            target_name: n.name,
            confidence: r.confidence
        })[..$max_edges] AS related
        """,
        {"chunk_id": chunk_id, "max_edges": max_edges},
    ).single()
    related = related_row["related"] if related_row and related_row["related"] else []

    return {"mentions": mentions, "related": related}


def maybe_enrich_with_neo4j(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
    if args.no_graph:
        return
    if not args.neo4j_password:
        raise ValueError("Neo4j password missing. Provide --neo4j-password or set NEO4J_PASSWORD, or use --no-graph.")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_database) as session:
            for row in rows:
                row["graph"] = fetch_graph_context_for_chunk(session, row["chunk_id"], args.max_graph_edges)
    finally:
        driver.close()


def load_chunk_concepts(metadata_db: Path, graph_dir: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    chunk_to_concepts: Dict[str, List[str]] = {}
    concept_to_chunks: Dict[str, List[str]] = {}

    conn = sqlite3.connect(metadata_db)
    try:
        table_exists = conn.execute(
            """
            SELECT count(*) FROM sqlite_master
            WHERE type='table' AND name='chunk_concept_map'
            """
        ).fetchone()[0]
        if table_exists:
            rows = conn.execute("SELECT chunk_id, concept_id FROM chunk_concept_map").fetchall()
            for chunk_id, concept_id in rows:
                chid = str(chunk_id)
                cid = str(concept_id)
                chunk_to_concepts.setdefault(chid, []).append(cid)
                concept_to_chunks.setdefault(cid, []).append(chid)
    finally:
        conn.close()

    if chunk_to_concepts:
        return chunk_to_concepts, concept_to_chunks

    concepts_path = graph_dir / "concepts.jsonl"
    if not concepts_path.exists():
        raise FileNotFoundError(f"Missing chunk_concept_map table and fallback concepts file: {concepts_path}")
    for rec in read_jsonl(concepts_path):
        cid = str(rec.get("concept_id", ""))
        if not cid:
            continue
        for chunk_id in rec.get("mention_chunk_ids", []):
            chid = str(chunk_id)
            chunk_to_concepts.setdefault(chid, []).append(cid)
            concept_to_chunks.setdefault(cid, []).append(chid)
    return chunk_to_concepts, concept_to_chunks


def load_relations(graph_dir: Path) -> Dict[str, List[Tuple[str, float]]]:
    relations_path = graph_dir / "relations.jsonl"
    if not relations_path.exists():
        return {}
    adj: Dict[str, List[Tuple[str, float]]] = {}
    for r in read_jsonl(relations_path):
        src = str(r.get("source_concept_id", ""))
        tgt = str(r.get("target_concept_id", ""))
        if not src or not tgt:
            continue
        conf = float(r.get("confidence", 0.0))
        adj.setdefault(src, []).append((tgt, conf))
        adj.setdefault(tgt, []).append((src, conf))
    return adj


def build_graph_scores(
    seed_ids: List[str],
    seed_scores: Dict[str, float],
    chunk_to_concepts: Dict[str, List[str]],
    concept_to_chunks: Dict[str, List[str]],
    relations: Dict[str, List[Tuple[str, float]]],
    relation_decay: float,
    concept_boost_limit: int,
    max_chunks_per_concept: int,
) -> Tuple[Dict[str, float], List[str]]:
    concept_scores: Dict[str, float] = {}
    for chunk_id in seed_ids:
        w = max(0.0, seed_scores.get(chunk_id, 0.0))
        for cid in chunk_to_concepts.get(chunk_id, []):
            concept_scores[cid] = concept_scores.get(cid, 0.0) + w

    propagated = dict(concept_scores)
    for src, src_score in concept_scores.items():
        for tgt, conf in relations.get(src, []):
            propagated[tgt] = propagated.get(tgt, 0.0) + src_score * max(0.0, conf) * relation_decay

    top_concepts = sorted(propagated.items(), key=lambda x: x[1], reverse=True)[:concept_boost_limit]
    candidate_chunks = set(seed_ids)
    for cid, _ in top_concepts:
        for chunk_id in concept_to_chunks.get(cid, [])[:max_chunks_per_concept]:
            candidate_chunks.add(chunk_id)

    chunk_graph_score: Dict[str, float] = {}
    for chunk_id in candidate_chunks:
        score = 0.0
        for cid in chunk_to_concepts.get(chunk_id, []):
            score += propagated.get(cid, 0.0)
        chunk_graph_score[chunk_id] = score
    return chunk_graph_score, sorted(candidate_chunks)


def build_annotate_results(
    args: argparse.Namespace,
    id_map: List[str],
    scores: np.ndarray,
    metadata_db: Path,
) -> List[Dict[str, Any]]:
    idx = top_k_indices(scores, args.top_k)
    rows = [
        {
            "rank": rank,
            "vector_row_index": int(i),
            "chunk_id": id_map[int(i)],
            "score": float(scores[int(i)]),
            "vector_score": float(scores[int(i)]),
        }
        for rank, i in enumerate(idx, start=1)
    ]

    chunk_meta = fetch_chunk_metadata(metadata_db, [x["chunk_id"] for x in rows])
    out: List[Dict[str, Any]] = []
    for item in rows:
        row = {**item, **chunk_meta.get(item["chunk_id"], {})}
        out.append(row)
    return out


def build_lightgraph_results(
    args: argparse.Namespace,
    id_map: List[str],
    scores: np.ndarray,
    metadata_db: Path,
    chunk_to_concepts: Dict[str, List[str]],
    concept_to_chunks: Dict[str, List[str]],
    relations: Dict[str, List[Tuple[str, float]]],
) -> List[Dict[str, Any]]:
    id_to_row = {cid: idx for idx, cid in enumerate(id_map)}
    seed_idx = top_k_indices(scores, args.seed_top_k)
    seed_ids = [id_map[int(i)] for i in seed_idx]

    if args.no_graph:
        ranked_ids = seed_ids[: args.top_k]
        graph_norm = {cid: 0.0 for cid in ranked_ids}
        final_norm = {cid: float(scores[id_to_row[cid]]) for cid in ranked_ids if cid in id_to_row}
    else:
        seed_scores = {id_map[int(i)]: float(scores[int(i)]) for i in seed_idx}
        graph_scores, candidates = build_graph_scores(
            seed_ids=seed_ids,
            seed_scores=seed_scores,
            chunk_to_concepts=chunk_to_concepts,
            concept_to_chunks=concept_to_chunks,
            relations=relations,
            relation_decay=args.relation_decay,
            concept_boost_limit=args.concept_boost_limit,
            max_chunks_per_concept=args.max_chunks_per_concept,
        )

        vector_raw = {cid: float(scores[id_to_row[cid]]) for cid in candidates if cid in id_to_row}
        v_norm = minmax_norm(vector_raw)
        g_norm = minmax_norm(graph_scores)
        combined = {cid: args.alpha * v_norm.get(cid, 0.0) + args.beta * g_norm.get(cid, 0.0) for cid in vector_raw}
        ranked_ids = [cid for cid, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[: args.top_k]]
        graph_norm = g_norm
        final_norm = combined

    chunk_meta = fetch_chunk_metadata(metadata_db, ranked_ids)
    results: List[Dict[str, Any]] = []
    for rank, cid in enumerate(ranked_ids, start=1):
        vec_score = float(scores[id_to_row[cid]]) if cid in id_to_row else 0.0
        row_idx = int(id_to_row[cid]) if cid in id_to_row else -1
        row = {
            "rank": rank,
            "vector_row_index": row_idx,
            "chunk_id": cid,
            "score": float(final_norm.get(cid, vec_score)),
            "final_score": float(final_norm.get(cid, vec_score)),
            "vector_score": vec_score,
            "graph_score": float(graph_norm.get(cid, 0.0)),
        }
        row.update(chunk_meta.get(cid, {}))
        results.append(row)
    return results


def run() -> None:
    args = parse_args()
    matrix, id_map, manifest = load_vector_artifacts(args.vector_dir)

    metadata_db = args.metadata_db or (args.vector_dir / "metadata.sqlite")
    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    embedder = get_embedder(args)
    qvec = embedder.encode([args.query])[0]

    if qvec.shape[0] != matrix.shape[1]:
        raise RuntimeError(
            f"Embedding dimension mismatch: query={qvec.shape[0]}, index={matrix.shape[1]}. "
            "Use matching --embedding-model and --embedding-dim for this vector index."
        )

    scores = vector_scores(matrix, qvec)

    if args.mode == "annotate":
        results = build_annotate_results(args=args, id_map=id_map, scores=scores, metadata_db=metadata_db)
        maybe_enrich_with_neo4j(args, results)
    else:
        chunk_to_concepts, concept_to_chunks = load_chunk_concepts(metadata_db=metadata_db, graph_dir=args.graph_dir)
        relations = {} if args.no_graph else load_relations(args.graph_dir)
        results = build_lightgraph_results(
            args=args,
            id_map=id_map,
            scores=scores,
            metadata_db=metadata_db,
            chunk_to_concepts=chunk_to_concepts,
            concept_to_chunks=concept_to_chunks,
            relations=relations,
        )

    payload = {
        "query": args.query,
        "mode": args.mode,
        "top_k": args.top_k,
        "vector_index": str(args.vector_dir / "index.npy"),
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "index_embedding_model": manifest.get("embedding_model"),
        "index_embedding_dim": manifest.get("embedding_dim"),
        "graph_enabled": not args.no_graph,
        "results": results,
    }

    if args.mode == "rerank_lightgraph":
        payload["light_graph_config"] = {
            "graph_dir": str(args.graph_dir),
            "seed_top_k": args.seed_top_k,
            "alpha": args.alpha,
            "beta": args.beta,
            "relation_decay": args.relation_decay,
            "concept_boost_limit": args.concept_boost_limit,
            "max_chunks_per_concept": args.max_chunks_per_concept,
        }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
