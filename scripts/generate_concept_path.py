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

from src.retrieval_pipeline.common import read_jsonl, utc_now_iso


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
    parser = argparse.ArgumentParser(description="Generate concept learning path from vector + concept graph.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--vector-dir", type=Path, default=Path("storage/vector"))
    parser.add_argument("--graph-dir", type=Path, default=Path("storage/graph"))
    parser.add_argument("--metadata-db", type=Path, default=None, help="Defaults to <vector-dir>/metadata.sqlite")
    parser.add_argument("--seed-top-k", type=int, default=20, help="Vector top-k chunks used as seed.")
    parser.add_argument("--max-concepts", type=int, default=8)
    parser.add_argument("--max-support-chunks", type=int, default=3)
    parser.add_argument("--expand-hops", type=int, default=2)
    parser.add_argument("--relation-decay", type=float, default=0.45)
    parser.add_argument("--output-json", type=Path, default=Path("storage/retrieval/concept_path.json"))

    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "sentence_transformers", "openai"],
        default="hash",
    )
    parser.add_argument("--embedding-model", default="hash-v1")
    parser.add_argument("--embedding-dim", type=int, default=384)
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

    matrix = np.load(index_path).astype(np.float32)
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if matrix.shape[0] != len(id_map):
        raise RuntimeError("index.npy row count does not match id_map length.")
    return matrix, [str(x) for x in id_map], manifest


def vector_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
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
        raise FileNotFoundError(f"Missing chunk_concept_map and fallback concepts file: {concepts_path}")
    for rec in read_jsonl(concepts_path):
        cid = str(rec.get("concept_id", ""))
        if not cid:
            continue
        for chunk_id in rec.get("mention_chunk_ids", []):
            chid = str(chunk_id)
            chunk_to_concepts.setdefault(chid, []).append(cid)
            concept_to_chunks.setdefault(cid, []).append(chid)
    return chunk_to_concepts, concept_to_chunks


def load_concept_catalog(graph_dir: Path) -> Dict[str, Dict[str, Any]]:
    concepts_path = graph_dir / "concepts.jsonl"
    if not concepts_path.exists():
        raise FileNotFoundError(f"Concept file not found: {concepts_path}")
    out: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(concepts_path):
        cid = str(rec.get("concept_id", ""))
        if not cid:
            continue
        out[cid] = {
            "concept_id": cid,
            "name": str(rec.get("name", cid)),
            "type": str(rec.get("type", "concept")),
            "description": str(rec.get("description", "")),
        }
    return out


def load_relation_graph(
    graph_dir: Path,
) -> Tuple[Dict[str, List[Tuple[str, float]]], List[Tuple[str, str, str, float]]]:
    relations_path = graph_dir / "relations.jsonl"
    if not relations_path.exists():
        return {}, []
    undirected: Dict[str, List[Tuple[str, float]]] = {}
    relation_rows: List[Tuple[str, str, str, float]] = []
    for rec in read_jsonl(relations_path):
        src = str(rec.get("source_concept_id", ""))
        tgt = str(rec.get("target_concept_id", ""))
        rel = str(rec.get("relation", ""))
        conf = float(rec.get("confidence", 0.0))
        if not src or not tgt:
            continue
        undirected.setdefault(src, []).append((tgt, conf))
        undirected.setdefault(tgt, []).append((src, conf))
        relation_rows.append((src, rel, tgt, conf))
    return undirected, relation_rows


def expand_concept_scores(
    seed_scores: Dict[str, float],
    neighbors: Dict[str, List[Tuple[str, float]]],
    hops: int,
    decay: float,
) -> Dict[str, float]:
    total = dict(seed_scores)
    frontier = dict(seed_scores)
    for hop in range(1, hops + 1):
        nxt: Dict[str, float] = {}
        hop_decay = decay**hop
        for src, score in frontier.items():
            if score <= 0.0:
                continue
            for tgt, conf in neighbors.get(src, []):
                inc = score * max(0.0, conf) * hop_decay
                if inc <= 0.0:
                    continue
                nxt[tgt] = nxt.get(tgt, 0.0) + inc
                total[tgt] = total.get(tgt, 0.0) + inc
        frontier = nxt
        if not frontier:
            break
    return total


def build_dependency_edges(
    relation_rows: List[Tuple[str, str, str, float]],
    candidates: set[str],
) -> List[Tuple[str, str, str, float]]:
    dep_edges: List[Tuple[str, str, str, float]] = []
    dep_relations = {"depends_on", "part_of", "is_a"}
    for src, rel, tgt, conf in relation_rows:
        if rel not in dep_relations:
            continue
        if src not in candidates or tgt not in candidates:
            continue
        # src depends_on/part_of/is_a tgt => tgt should come before src in learning order.
        dep_edges.append((tgt, src, rel, conf))
    return dep_edges


def order_concepts_by_dependency(
    candidates: List[str],
    concept_scores: Dict[str, float],
    dep_edges: List[Tuple[str, str, str, float]],
) -> List[str]:
    nodes = set(candidates)
    indegree = {n: 0 for n in nodes}
    out_edges: Dict[str, List[str]] = {n: [] for n in nodes}
    for u, v, _, _ in dep_edges:
        if u in nodes and v in nodes and v not in out_edges[u]:
            out_edges[u].append(v)
            indegree[v] += 1

    available = [n for n in nodes if indegree[n] == 0]
    available.sort(key=lambda x: concept_scores.get(x, 0.0), reverse=True)

    ordered: List[str] = []
    while available:
        n = available.pop(0)
        ordered.append(n)
        newly = []
        for v in out_edges.get(n, []):
            indegree[v] -= 1
            if indegree[v] == 0:
                newly.append(v)
        if newly:
            available.extend(newly)
            available.sort(key=lambda x: concept_scores.get(x, 0.0), reverse=True)

    if len(ordered) < len(nodes):
        remaining = [n for n in nodes if n not in set(ordered)]
        remaining.sort(key=lambda x: concept_scores.get(x, 0.0), reverse=True)
        ordered.extend(remaining)

    return ordered


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
            """
            ,
            chunk_ids,
        ).fetchall()
    finally:
        conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[r[0]] = {
            "source_name": r[1],
            "page_start": r[2],
            "page_end": r[3],
            "char_count": r[4],
            "token_estimate": r[5],
            "chunk_preview": (r[6] or "")[:220],
        }
    return out


def run() -> None:
    args = parse_args()
    matrix, id_map, manifest = load_vector_artifacts(args.vector_dir)
    id_to_row = {cid: idx for idx, cid in enumerate(id_map)}

    metadata_db = args.metadata_db or (args.vector_dir / "metadata.sqlite")
    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    embedder = get_embedder(args)
    qvec = embedder.encode([args.query])[0]
    if qvec.shape[0] != matrix.shape[1]:
        raise RuntimeError(
            f"Embedding dimension mismatch: query={qvec.shape[0]}, index={matrix.shape[1]}. "
            "Use matching embedding settings."
        )

    scores = vector_scores(matrix, qvec)
    seed_idx = top_k_indices(scores, args.seed_top_k)
    seed_chunks = [id_map[int(i)] for i in seed_idx]
    seed_chunk_scores = {id_map[int(i)]: float(scores[int(i)]) for i in seed_idx}

    chunk_to_concepts, concept_to_chunks = load_chunk_concepts(metadata_db=metadata_db, graph_dir=args.graph_dir)
    concept_catalog = load_concept_catalog(args.graph_dir)
    undirected_neighbors, relation_rows = load_relation_graph(args.graph_dir)

    seed_concept_scores: Dict[str, float] = {}
    for chunk_id in seed_chunks:
        base = max(0.0, seed_chunk_scores.get(chunk_id, 0.0))
        for cid in chunk_to_concepts.get(chunk_id, []):
            seed_concept_scores[cid] = seed_concept_scores.get(cid, 0.0) + base

    expanded_scores = expand_concept_scores(
        seed_scores=seed_concept_scores,
        neighbors=undirected_neighbors,
        hops=args.expand_hops,
        decay=args.relation_decay,
    )
    ranked_concepts = sorted(expanded_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_ids = [cid for cid, _ in ranked_concepts[: max(args.max_concepts * 3, args.max_concepts)]]
    candidate_set = set(candidate_ids)

    dep_edges = build_dependency_edges(relation_rows=relation_rows, candidates=candidate_set)
    ordered = order_concepts_by_dependency(
        candidates=candidate_ids,
        concept_scores=expanded_scores,
        dep_edges=dep_edges,
    )[: args.max_concepts]

    chunk_meta = fetch_chunk_metadata(metadata_db, seed_chunks)
    support_chunk_ids: set[str] = set()
    for cid in ordered:
        support_chunk_ids.update(concept_to_chunks.get(cid, []))
    support_chunk_meta = fetch_chunk_metadata(metadata_db, sorted(support_chunk_ids))
    steps = []
    for idx, cid in enumerate(ordered, start=1):
        info = concept_catalog.get(
            cid,
            {"concept_id": cid, "name": cid, "type": "concept", "description": ""},
        )
        support = concept_to_chunks.get(cid, [])
        support = sorted(
            support,
            key=lambda ch: seed_chunk_scores.get(ch, float(scores[id_to_row[ch]]) if ch in id_to_row else -1.0),
            reverse=True,
        )[: args.max_support_chunks]
        prereqs = [u for (u, v, _, _) in dep_edges if v == cid and u in ordered]
        support_rows = []
        page_starts = []
        page_ends = []
        for chunk_id in support:
            meta = support_chunk_meta.get(chunk_id, chunk_meta.get(chunk_id, {}))
            page_start = meta.get("page_start")
            page_end = meta.get("page_end")
            if isinstance(page_start, int):
                page_starts.append(page_start)
            if isinstance(page_end, int):
                page_ends.append(page_end)
            support_rows.append(
                {
                    "chunk_id": chunk_id,
                    "source_name": meta.get("source_name"),
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_preview": meta.get("chunk_preview"),
                }
            )

        concept_page_span = {
            "page_start_min": min(page_starts) if page_starts else None,
            "page_end_max": max(page_ends) if page_ends else None,
        }
        entry = support_rows[0] if support_rows else {}

        steps.append(
            {
                "step": idx,
                "concept_id": cid,
                "concept_name": info["name"],
                "concept_type": info["type"],
                "score": float(expanded_scores.get(cid, 0.0)),
                "prerequisites": sorted(set(prereqs)),
                "concept_page_span": concept_page_span,
                "recommended_chunk_id": entry.get("chunk_id"),
                "recommended_source_name": entry.get("source_name"),
                "recommended_start_page": entry.get("page_start"),
                "recommended_end_page": entry.get("page_end"),
                "support_chunks": support_rows,
            }
        )

    seed_rows = []
    for rank, chunk_id in enumerate(seed_chunks, start=1):
        row = {
            "rank": rank,
            "chunk_id": chunk_id,
            "vector_score": seed_chunk_scores.get(chunk_id, 0.0),
            "concepts": chunk_to_concepts.get(chunk_id, []),
        }
        row.update(chunk_meta.get(chunk_id, {}))
        seed_rows.append(row)

    payload = {
        "created_at": utc_now_iso(),
        "query": args.query,
        "vector_dir": str(args.vector_dir),
        "graph_dir": str(args.graph_dir),
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "index_embedding_model": manifest.get("embedding_model"),
        "index_embedding_dim": manifest.get("embedding_dim"),
        "config": {
            "seed_top_k": args.seed_top_k,
            "max_concepts": args.max_concepts,
            "max_support_chunks": args.max_support_chunks,
            "expand_hops": args.expand_hops,
            "relation_decay": args.relation_decay,
        },
        "seed_chunks": seed_rows,
        "concept_path": steps,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("Concept path generation complete.")
    print(f"Query: {args.query}")
    print(f"Path length: {len(steps)}")
    for s in steps:
        print(f"  {s['step']:>2}. {s['concept_name']} ({s['concept_id']}) score={s['score']:.4f}")
    print(f"Output: {args.output_json}")


if __name__ == "__main__":
    run()
