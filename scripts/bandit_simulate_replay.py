#!/usr/bin/env python3
import argparse
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.bandit import Arm, LinUCB, epsilon_greedy_choice, retrieval_greedy_index


DEFAULT_QUERIES = [
    "How does CT differ from MRI?",
    "What is Fourier transform used for in imaging?",
    "How does ultrasound imaging work?",
    "What are benefits of X-ray imaging?",
    "Explain convolution in image processing",
]


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
        vec = np.asarray(values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector_from_text(t) for t in texts]).astype(np.float32)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tokenize(text: str) -> set[str]:
    out = []
    token = []
    for ch in text.lower():
        if ch.isalnum():
            token.append(ch)
        else:
            if token:
                out.append("".join(token))
                token = []
    if token:
        out.append("".join(token))
    return set(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate replay log with propensities and rewards.")
    parser.add_argument("--vector-dir", type=Path, default=Path("storage/vector"))
    parser.add_argument("--events-out", type=Path, default=Path("storage/bandit/events.jsonl"))
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--policy", choices=["retrieval_epsilon", "linucb_epsilon"], default="retrieval_epsilon")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--linucb-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--queries-file", type=Path, default=None)
    return parser.parse_args()


def load_queries(path: Path | None) -> List[str]:
    if path is None:
        return list(DEFAULT_QUERIES)
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def load_vector_artifacts(vector_dir: Path) -> tuple[np.ndarray, List[str], sqlite3.Connection]:
    matrix = np.load(vector_dir / "index.npy").astype(np.float32)
    id_map = json.loads((vector_dir / "id_map.json").read_text(encoding="utf-8"))
    conn = sqlite3.connect(vector_dir / "metadata.sqlite")
    return matrix, [str(x) for x in id_map], conn


def get_chunk_rows(conn: sqlite3.Connection, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    placeholders = ",".join(["?"] * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT chunk_id, chunk_text, char_count, token_estimate
        FROM chunk_embeddings
        WHERE chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    return {
        r[0]: {
            "chunk_text": r[1] or "",
            "char_count": int(r[2] or 0),
            "token_estimate": int(r[3] or 0),
        }
        for r in rows
    }


def vector_top_k(matrix: np.ndarray, id_map: List[str], query_vec: np.ndarray, k: int) -> List[Dict[str, Any]]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix_n = matrix / norms
    qn = query_vec / (np.linalg.norm(query_vec) or 1.0)

    scores = matrix_n @ qn
    k = min(k, len(id_map))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [
        {
            "vector_row_index": int(i),
            "chunk_id": id_map[i],
            "retrieval_score": float(scores[i]),
            "rank": rank,
        }
        for rank, i in enumerate(idx, start=1)
    ]


def build_features(query: str, candidate: Dict[str, Any], chunk_meta: Dict[str, Any]) -> np.ndarray:
    q_tokens = tokenize(query)
    c_tokens = tokenize(chunk_meta.get("chunk_text", "")[:800])
    overlap = len(q_tokens.intersection(c_tokens))
    overlap_ratio = overlap / max(1, len(q_tokens))

    retrieval_score = float(candidate["retrieval_score"])
    rank_feature = 1.0 / float(candidate["rank"])
    char_norm = min(1.0, float(chunk_meta.get("char_count", 0)) / 3000.0)
    token_norm = min(1.0, float(chunk_meta.get("token_estimate", 0)) / 800.0)

    return np.asarray([retrieval_score, rank_feature, char_norm, token_norm, overlap_ratio], dtype=np.float64)


def simulate_reward(query: str, chunk_text: str, rng: np.random.Generator) -> float:
    q = tokenize(query)
    c = tokenize(chunk_text[:1500])
    overlap = len(q.intersection(c))
    base = min(1.0, overlap / max(1, len(q)))
    noise = float(rng.normal(0.0, 0.05))
    reward = max(0.0, min(1.0, base + noise))
    return reward


def run() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    queries = load_queries(args.queries_file)
    if not queries:
        raise ValueError("No queries available for simulation.")

    matrix, id_map, conn = load_vector_artifacts(args.vector_dir)
    embedder = HashEmbedder(dim=matrix.shape[1])
    model = LinUCB(dim=5, alpha=args.linucb_alpha)

    args.events_out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    total_reward = 0.0

    with args.events_out.open("w", encoding="utf-8") as f:
        for t in range(args.rounds):
            query = queries[t % len(queries)]
            qvec = embedder.encode([query])[0]
            cands = vector_top_k(matrix, id_map, qvec, args.top_k)
            chunk_meta = get_chunk_rows(conn, [x["chunk_id"] for x in cands])

            arms: List[Arm] = []
            serialized_candidates: List[Dict[str, Any]] = []
            for c in cands:
                meta = chunk_meta[c["chunk_id"]]
                feat = build_features(query, c, meta)
                arms.append(Arm(chunk_id=c["chunk_id"], feature=feat, retrieval_score=c["retrieval_score"]))
                serialized_candidates.append(
                    {
                        "chunk_id": c["chunk_id"],
                        "retrieval_score": c["retrieval_score"],
                        "rank": c["rank"],
                        "feature": feat.tolist(),
                    }
                )

            if args.policy == "retrieval_epsilon":
                best_idx = retrieval_greedy_index(arms)
            else:
                best_idx = model.choose_index([a.feature for a in arms])

            chosen_idx, propensity = epsilon_greedy_choice(best_idx, len(arms), args.epsilon, rng)
            chosen = arms[chosen_idx]

            reward = simulate_reward(query, chunk_meta[chosen.chunk_id]["chunk_text"], rng)
            total_reward += reward

            if args.policy == "linucb_epsilon":
                model.update(chosen.feature, reward)

            event = {
                "event_id": str(uuid.uuid4()),
                "ts": now_iso(),
                "round": t,
                "query": query,
                "policy_name": args.policy,
                "epsilon": args.epsilon,
                "candidate_arms": serialized_candidates,
                "chosen_arm": {
                    "chunk_id": chosen.chunk_id,
                    "arm_index": chosen_idx,
                    "feature": chosen.feature.tolist(),
                    "propensity": propensity,
                },
                "reward": reward,
            }
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
            n_written += 1

    conn.close()

    print("Replay simulation complete.")
    print(f"Events written: {n_written}")
    print(f"Average reward: {total_reward / max(1, n_written):.4f}")
    print(f"Output: {args.events_out}")


if __name__ == "__main__":
    run()
