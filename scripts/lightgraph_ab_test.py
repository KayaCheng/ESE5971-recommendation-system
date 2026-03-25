#!/usr/bin/env python3
import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.common import read_jsonl, utc_now_iso


DEFAULT_QUERIES = [
    "How does CT differ from MRI?",
    "What is Fourier transform used for in imaging?",
    "How does ultrasound imaging work?",
    "What are benefits of X-ray imaging?",
    "Explain convolution in image processing",
]


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
    parser = argparse.ArgumentParser(description="Quick local A/B test: vector-only vs vector+light-graph.")
    parser.add_argument("--vector-dir", type=Path, default=Path("storage/vector"))
    parser.add_argument("--graph-dir", type=Path, default=Path("storage/graph"))
    parser.add_argument("--metadata-db", type=Path, default=None, help="Defaults to <vector-dir>/metadata.sqlite")
    parser.add_argument("--queries-file", type=Path, default=None, help="One query per line.")
    parser.add_argument(
        "--qrels-json",
        type=Path,
        default=None,
        help="Optional JSON file: [{'query': str, 'relevant_chunk_ids': [str,...]}].",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-top-k", type=int, default=20)

    parser.add_argument("--alpha", type=float, default=0.70, help="Weight for vector score.")
    parser.add_argument("--beta", type=float, default=0.30, help="Weight for graph score.")
    parser.add_argument("--relation-decay", type=float, default=0.35)
    parser.add_argument("--concept-boost-limit", type=int, default=16)
    parser.add_argument("--max-chunks-per-concept", type=int, default=40)

    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "sentence_transformers", "openai"],
        default="hash",
    )
    parser.add_argument("--embedding-model", default="hash-v1")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--output-json", type=Path, default=Path("storage/retrieval/lightgraph_ab_report.json"))
    return parser.parse_args()


def get_embedder(args: argparse.Namespace):
    if args.embedding_backend == "hash":
        return HashEmbedder(dim=args.embedding_dim)
    if args.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=args.embedding_model)
    if args.embedding_backend == "openai":
        import os

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
                cid = str(concept_id)
                chid = str(chunk_id)
                chunk_to_concepts.setdefault(chid, []).append(cid)
                concept_to_chunks.setdefault(cid, []).append(chid)
    finally:
        conn.close()

    if chunk_to_concepts:
        return chunk_to_concepts, concept_to_chunks

    concepts_path = graph_dir / "concepts.jsonl"
    if not concepts_path.exists():
        raise FileNotFoundError(
            f"chunk_concept_map not found in sqlite and fallback concepts file missing: {concepts_path}"
        )
    for rec in read_jsonl(concepts_path):
        cid = str(rec["concept_id"])
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


def load_concept_aliases(graph_dir: Path) -> Dict[str, List[str]]:
    concepts_path = graph_dir / "concepts.jsonl"
    aliases: Dict[str, List[str]] = {}
    if not concepts_path.exists():
        return aliases
    for rec in read_jsonl(concepts_path):
        cid = str(rec["concept_id"])
        terms = [str(rec.get("name", "")).strip().lower()]
        terms.extend([str(x).strip().lower() for x in rec.get("aliases", [])])
        terms = [t for t in terms if t]
        aliases[cid] = sorted(set(terms), key=len, reverse=True)
    return aliases


def vector_scores(matrix: np.ndarray, query_vec: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix_n = matrix / norms
    q = l2_normalize(query_vec.astype(np.float32))
    return matrix_n @ q


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(k, scores.size)
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def rank_vector_only(id_map: List[str], all_scores: np.ndarray, top_k: int) -> List[str]:
    return [id_map[int(i)] for i in top_k_indices(all_scores, top_k)]


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


def rank_hybrid(
    id_map: List[str],
    id_to_row: Dict[str, int],
    all_scores: np.ndarray,
    chunk_to_concepts: Dict[str, List[str]],
    concept_to_chunks: Dict[str, List[str]],
    relations: Dict[str, List[Tuple[str, float]]],
    seed_top_k: int,
    top_k: int,
    alpha: float,
    beta: float,
    relation_decay: float,
    concept_boost_limit: int,
    max_chunks_per_concept: int,
) -> List[str]:
    seed_idx = top_k_indices(all_scores, seed_top_k)
    seed_ids = [id_map[int(i)] for i in seed_idx]
    seed_scores = {id_map[int(i)]: float(all_scores[int(i)]) for i in seed_idx}

    chunk_graph_score, candidates = build_graph_scores(
        seed_ids=seed_ids,
        seed_scores=seed_scores,
        chunk_to_concepts=chunk_to_concepts,
        concept_to_chunks=concept_to_chunks,
        relations=relations,
        relation_decay=relation_decay,
        concept_boost_limit=concept_boost_limit,
        max_chunks_per_concept=max_chunks_per_concept,
    )

    candidate_v = {cid: float(all_scores[id_to_row[cid]]) for cid in candidates if cid in id_to_row}
    v_norm = minmax_norm(candidate_v)
    g_norm = minmax_norm(chunk_graph_score)

    combined = {}
    for cid in candidates:
        combined[cid] = alpha * v_norm.get(cid, 0.0) + beta * g_norm.get(cid, 0.0)
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:top_k]]


def read_queries(path: Path | None) -> List[str]:
    if path is None:
        return list(DEFAULT_QUERIES)
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def read_qrels(path: Path | None) -> Dict[str, set[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, set[str]] = {}
    if not isinstance(payload, list):
        raise ValueError("--qrels-json must be a JSON list.")
    for row in payload:
        q = str(row.get("query", "")).strip()
        rel = {str(x) for x in row.get("relevant_chunk_ids", [])}
        if q and rel:
            out[q] = rel
    return out


def infer_proxy_relevant_set(query: str, concept_aliases: Dict[str, List[str]], concept_to_chunks: Dict[str, List[str]]) -> set[str]:
    q = query.lower().strip()
    matched: set[str] = set()
    for cid, aliases in concept_aliases.items():
        if any(alias in q for alias in aliases):
            matched.add(cid)
    rel_chunks: set[str] = set()
    for cid in matched:
        rel_chunks.update(concept_to_chunks.get(cid, []))
    return rel_chunks


def metrics_at_k(ranked: List[str], relevant: set[str], k: int) -> Dict[str, float]:
    if not relevant:
        return {"MRR": 0.0, "Recall": 0.0, "nDCG": 0.0}
    top = ranked[:k]
    hits = [1 if cid in relevant else 0 for cid in top]
    recall = float(sum(hits) / len(relevant))

    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / i
            break

    dcg = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = [1] * min(k, len(relevant))
    idcg = 0.0
    for i, h in enumerate(ideal_hits, start=1):
        if h:
            idcg += 1.0 / math.log2(i + 1)
    ndcg = 0.0 if idcg == 0.0 else float(dcg / idcg)
    return {"MRR": rr, "Recall": recall, "nDCG": ndcg}


def aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"MRR": 0.0, "Recall": 0.0, "nDCG": 0.0}
    return {
        "MRR": float(np.mean([m["MRR"] for m in metrics])),
        "Recall": float(np.mean([m["Recall"] for m in metrics])),
        "nDCG": float(np.mean([m["nDCG"] for m in metrics])),
    }


def run() -> None:
    args = parse_args()
    queries = read_queries(args.queries_file)
    if not queries:
        raise ValueError("No queries provided.")

    matrix, id_map, manifest = load_vector_artifacts(args.vector_dir)
    metadata_db = args.metadata_db or (args.vector_dir / "metadata.sqlite")
    if not metadata_db.exists():
        raise FileNotFoundError(f"Metadata DB not found: {metadata_db}")

    chunk_to_concepts, concept_to_chunks = load_chunk_concepts(metadata_db=metadata_db, graph_dir=args.graph_dir)
    relations = load_relations(args.graph_dir)
    concept_aliases = load_concept_aliases(args.graph_dir)
    qrels = read_qrels(args.qrels_json)

    embedder = get_embedder(args)
    query_vecs = embedder.encode(queries)
    if query_vecs.shape[1] != matrix.shape[1]:
        raise RuntimeError(
            f"Embedding dimension mismatch: query={query_vecs.shape[1]}, index={matrix.shape[1]}. "
            "Use matching embedding settings."
        )

    detail_rows = []
    vector_metrics_rows = []
    hybrid_metrics_rows = []
    using_proxy = args.qrels_json is None
    id_to_row = {cid: idx for idx, cid in enumerate(id_map)}

    for query, qvec in zip(queries, query_vecs):
        scores = vector_scores(matrix, qvec)
        vector_ranked = rank_vector_only(id_map=id_map, all_scores=scores, top_k=args.top_k)
        hybrid_ranked = rank_hybrid(
            id_map=id_map,
            id_to_row=id_to_row,
            all_scores=scores,
            chunk_to_concepts=chunk_to_concepts,
            concept_to_chunks=concept_to_chunks,
            relations=relations,
            seed_top_k=args.seed_top_k,
            top_k=args.top_k,
            alpha=args.alpha,
            beta=args.beta,
            relation_decay=args.relation_decay,
            concept_boost_limit=args.concept_boost_limit,
            max_chunks_per_concept=args.max_chunks_per_concept,
        )

        if query in qrels:
            relevant = qrels[query]
        else:
            relevant = infer_proxy_relevant_set(query, concept_aliases=concept_aliases, concept_to_chunks=concept_to_chunks)

        vm = metrics_at_k(vector_ranked, relevant, args.top_k)
        hm = metrics_at_k(hybrid_ranked, relevant, args.top_k)
        vector_metrics_rows.append(vm)
        hybrid_metrics_rows.append(hm)

        detail_rows.append(
            {
                "query": query,
                "relevant_count": len(relevant),
                "vector_only_top_k": vector_ranked,
                "hybrid_light_graph_top_k": hybrid_ranked,
                "vector_only_metrics": vm,
                "hybrid_metrics": hm,
            }
        )

    vector_avg = aggregate(vector_metrics_rows)
    hybrid_avg = aggregate(hybrid_metrics_rows)

    report = {
        "created_at": utc_now_iso(),
        "mode": "qrels" if not using_proxy else "proxy_concept_match",
        "top_k": args.top_k,
        "seed_top_k": args.seed_top_k,
        "weights": {
            "alpha_vector": args.alpha,
            "beta_graph": args.beta,
            "relation_decay": args.relation_decay,
        },
        "embedding_backend": args.embedding_backend,
        "embedding_model": args.embedding_model,
        "index_embedding_model": manifest.get("embedding_model"),
        "index_embedding_dim": manifest.get("embedding_dim"),
        "num_queries": len(queries),
        "aggregate": {
            "vector_only": vector_avg,
            "hybrid_light_graph": hybrid_avg,
            "delta_hybrid_minus_vector": {
                "MRR": hybrid_avg["MRR"] - vector_avg["MRR"],
                "Recall": hybrid_avg["Recall"] - vector_avg["Recall"],
                "nDCG": hybrid_avg["nDCG"] - vector_avg["nDCG"],
            },
        },
        "queries": detail_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Light-graph A/B test complete.")
    print(f"Mode: {report['mode']}")
    print(f"Queries: {len(queries)}")
    print("Vector-only metrics:")
    print(
        f"  MRR={vector_avg['MRR']:.4f}, Recall={vector_avg['Recall']:.4f}, nDCG={vector_avg['nDCG']:.4f}"
    )
    print("Hybrid(light-graph) metrics:")
    print(
        f"  MRR={hybrid_avg['MRR']:.4f}, Recall={hybrid_avg['Recall']:.4f}, nDCG={hybrid_avg['nDCG']:.4f}"
    )
    d = report["aggregate"]["delta_hybrid_minus_vector"]
    print(f"Delta(hybrid-vector): MRR={d['MRR']:.4f}, Recall={d['Recall']:.4f}, nDCG={d['nDCG']:.4f}")
    print(f"Report: {args.output_json}")


if __name__ == "__main__":
    run()
