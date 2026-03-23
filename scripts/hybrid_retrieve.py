#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


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
    parser = argparse.ArgumentParser(description="Hybrid retrieval: vector recall + graph expansion.")
    parser.add_argument("--query", required=True, help="User query text.")
    parser.add_argument("--vector-dir", type=Path, default=Path("storage/vector"))
    parser.add_argument("--metadata-db", type=Path, default=None, help="Defaults to <vector-dir>/metadata.sqlite")
    parser.add_argument("--top-k", type=int, default=5)

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
    parser.add_argument("--no-graph", action="store_true", help="Skip Neo4j expansion and return vector-only results.")
    parser.add_argument("--max-graph-edges", type=int, default=8)

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


def load_vector_artifacts(vector_dir: Path) -> tuple[np.ndarray, List[str], Dict[str, Any]]:
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


def vector_search(matrix: np.ndarray, id_map: List[str], query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    if matrix.size == 0:
        return []

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix_n = matrix / norms

    q = l2_normalize(query_vec.astype(np.float32))
    scores = matrix_n @ q

    k = min(top_k, len(id_map))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    results = []
    for rank, i in enumerate(idx, start=1):
        results.append(
            {
                "rank": rank,
                "vector_row_index": int(i),
                "chunk_id": id_map[i],
                "score": float(scores[i]),
            }
        )
    return results


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


def maybe_enrich_with_graph(args: argparse.Namespace, rows: List[Dict[str, Any]]) -> None:
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

    base = vector_search(matrix, id_map, qvec, args.top_k)
    chunk_meta = fetch_chunk_metadata(metadata_db, [x["chunk_id"] for x in base])

    results: List[Dict[str, Any]] = []
    for item in base:
        chunk_id = item["chunk_id"]
        row = {**item, **chunk_meta.get(chunk_id, {})}
        results.append(row)

    maybe_enrich_with_graph(args, results)

    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "vector_index": str(args.vector_dir / "index.npy"),
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "index_embedding_model": manifest.get("embedding_model"),
        "index_embedding_dim": manifest.get("embedding_dim"),
        "graph_enabled": not args.no_graph,
        "results": results,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
