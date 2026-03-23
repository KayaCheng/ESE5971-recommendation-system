#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.common import content_hash, ensure_parent_dir, read_jsonl, utc_now_iso


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    chunk_text TEXT NOT NULL,
    char_count INTEGER,
    token_estimate INTEGER,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class ChunkRecord:
    chunk_id: str
    source_name: str
    page_start: int
    page_end: int
    chunk_text: str
    char_count: int
    token_estimate: int


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
        return l2_normalize(vec)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._vector_from_text(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies first."
            ) from exc
        self.model_name = model_name
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


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local vector storage from chunk JSONL.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks/mis1to66_chunks.jsonl"),
        help="Path to chunk JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("storage/vector"),
        help="Directory to store index and metadata.",
    )
    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "sentence_transformers", "openai"],
        default="hash",
        help="Embedding provider backend.",
    )
    parser.add_argument(
        "--embedding-model",
        default="hash-v1",
        help="Embedding model name for metadata/versioning.",
    )
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Recompute embeddings even if hash/model match.",
    )
    return parser.parse_args()


def load_chunks(path: Path) -> List[ChunkRecord]:
    raw = read_jsonl(path)
    chunks: List[ChunkRecord] = []
    for idx, item in enumerate(raw, start=1):
        missing = [k for k in ("chunk_id", "source_name", "page_start", "page_end", "chunk_text") if k not in item]
        if missing:
            raise ValueError(f"Record #{idx} missing required keys: {missing}")
        chunks.append(
            ChunkRecord(
                chunk_id=str(item["chunk_id"]),
                source_name=str(item["source_name"]),
                page_start=int(item["page_start"]),
                page_end=int(item["page_end"]),
                chunk_text=str(item["chunk_text"]),
                char_count=int(item.get("char_count", len(str(item["chunk_text"])))),
                token_estimate=int(item.get("token_estimate", max(1, len(str(item["chunk_text"])) // 4))),
            )
        )
    return chunks


def get_embedder(args: argparse.Namespace):
    if args.embedding_backend == "hash":
        return HashEmbedder(dim=args.embedding_dim)
    if args.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=args.embedding_model)
    if args.embedding_backend == "openai":
        return OpenAIEmbedder(
            model_name=args.embedding_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    raise ValueError(f"Unsupported embedding backend: {args.embedding_backend}")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()


def fetch_existing(conn: sqlite3.Connection) -> Dict[str, Dict[str, str | int]]:
    rows = conn.execute(
        "SELECT chunk_id, content_hash, embedding_model, embedding_dim FROM chunk_embeddings"
    ).fetchall()
    return {
        r[0]: {
            "content_hash": r[1],
            "embedding_model": r[2],
            "embedding_dim": int(r[3]),
        }
        for r in rows
    }


def upsert_rows(conn: sqlite3.Connection, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO chunk_embeddings (
            chunk_id, source_name, page_start, page_end, chunk_text,
            char_count, token_estimate, content_hash,
            embedding_model, embedding_dim, vector_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(chunk_id) DO UPDATE SET
            source_name=excluded.source_name,
            page_start=excluded.page_start,
            page_end=excluded.page_end,
            chunk_text=excluded.chunk_text,
            char_count=excluded.char_count,
            token_estimate=excluded.token_estimate,
            content_hash=excluded.content_hash,
            embedding_model=excluded.embedding_model,
            embedding_dim=excluded.embedding_dim,
            vector_json=excluded.vector_json,
            updated_at=excluded.updated_at;
        """,
        [
            (
                r["chunk_id"],
                r["source_name"],
                r["page_start"],
                r["page_end"],
                r["chunk_text"],
                r["char_count"],
                r["token_estimate"],
                r["content_hash"],
                r["embedding_model"],
                r["embedding_dim"],
                r["vector_json"],
                r["updated_at"],
            )
            for r in rows
        ],
    )
    conn.commit()


def build_numpy_index(conn: sqlite3.Connection, chunks: List[ChunkRecord], model: str, dim: int, output_dir: Path) -> None:
    vectors: List[np.ndarray] = []
    id_map: List[str] = []

    for chunk in chunks:
        row = conn.execute(
            """
            SELECT vector_json, embedding_model, embedding_dim
            FROM chunk_embeddings
            WHERE chunk_id = ?
            """,
            (chunk.chunk_id,),
        ).fetchone()

        if row is None:
            raise RuntimeError(f"Missing embedding row for chunk_id={chunk.chunk_id}")
        if row[1] != model or int(row[2]) != dim:
            raise RuntimeError(
                f"Embedding version mismatch for chunk_id={chunk.chunk_id}. "
                f"Expected model={model}, dim={dim}, got model={row[1]}, dim={row[2]}"
            )

        vec = np.asarray(json.loads(row[0]), dtype=np.float32)
        if vec.shape[0] != dim:
            raise RuntimeError(f"Vector dimension mismatch for chunk_id={chunk.chunk_id}")

        vectors.append(vec)
        id_map.append(chunk.chunk_id)

    matrix = np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, dim), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "index.npy", matrix)
    ensure_parent_dir(output_dir / "id_map.json")
    with (output_dir / "id_map.json").open("w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)


def run() -> None:
    args = parse_args()

    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {args.chunks_path}")

    chunks = load_chunks(args.chunks_path)
    if not chunks:
        raise ValueError("No chunks found in input JSONL.")

    embedder = get_embedder(args)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "metadata.sqlite"

    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        existing = fetch_existing(conn)

        recompute_count = 0
        skipped_count = 0

        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]

            embed_texts: List[str] = []
            embed_records: List[ChunkRecord] = []
            hashed: Dict[str, str] = {}

            for chunk in batch:
                h = content_hash(chunk.chunk_text)
                hashed[chunk.chunk_id] = h

                old = existing.get(chunk.chunk_id)
                same = (
                    old is not None
                    and old["content_hash"] == h
                    and old["embedding_model"] == args.embedding_model
                    and int(old["embedding_dim"]) == args.embedding_dim
                )

                if same and not args.force_reembed:
                    skipped_count += 1
                    continue

                embed_texts.append(chunk.chunk_text)
                embed_records.append(chunk)

            if not embed_texts:
                continue

            vectors = embedder.encode(embed_texts)
            if vectors.shape[1] != args.embedding_dim:
                raise RuntimeError(
                    f"Embedding dim mismatch: expected {args.embedding_dim}, got {vectors.shape[1]}"
                )

            now = utc_now_iso()
            rows = []
            for idx, chunk in enumerate(embed_records):
                vec = vectors[idx].astype(np.float32)
                rows.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_name": chunk.source_name,
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "chunk_text": chunk.chunk_text,
                        "char_count": chunk.char_count,
                        "token_estimate": chunk.token_estimate,
                        "content_hash": hashed[chunk.chunk_id],
                        "embedding_model": args.embedding_model,
                        "embedding_dim": args.embedding_dim,
                        "vector_json": json.dumps(vec.tolist(), ensure_ascii=False),
                        "updated_at": now,
                    }
                )
            upsert_rows(conn, rows)
            recompute_count += len(rows)

            # Refresh cache for following batches in same run
            for r in rows:
                existing[r["chunk_id"]] = {
                    "content_hash": r["content_hash"],
                    "embedding_model": r["embedding_model"],
                    "embedding_dim": r["embedding_dim"],
                }

        build_numpy_index(
            conn=conn,
            chunks=chunks,
            model=args.embedding_model,
            dim=args.embedding_dim,
            output_dir=output_dir,
        )

        manifest = {
            "created_at": utc_now_iso(),
            "chunks_path": str(args.chunks_path),
            "num_chunks": len(chunks),
            "embedding_backend": args.embedding_backend,
            "embedding_model": args.embedding_model,
            "embedding_dim": args.embedding_dim,
            "recomputed_embeddings": recompute_count,
            "skipped_embeddings": skipped_count,
            "index_file": str(output_dir / "index.npy"),
            "id_map_file": str(output_dir / "id_map.json"),
            "metadata_db": str(db_path),
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    finally:
        conn.close()

    print("Vector storage build complete.")
    print(f"Chunks: {len(chunks)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    run()
