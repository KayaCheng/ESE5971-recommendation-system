#!/usr/bin/env python3
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.common import read_jsonl, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind vector storage and graph storage by chunk_id.")
    parser.add_argument("--vector-dir", type=Path, default=Path("storage/vector"))
    parser.add_argument("--graph-dir", type=Path, default=Path("storage/graph"))
    parser.add_argument("--index-name", default="mis1to66_vector_index")

    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--skip-neo4j", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200)
    return parser.parse_args()


def batched(rows: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def build_mapping_rows(vector_dir: Path, index_name: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    id_map_path = vector_dir / "id_map.json"
    manifest_path = vector_dir / "manifest.json"
    if not id_map_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Missing vector artifacts: id_map.json and manifest.json are required.")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with id_map_path.open("r", encoding="utf-8") as f:
        id_map = json.load(f)

    rows = []
    now = utc_now_iso()
    for idx, chunk_id in enumerate(id_map):
        rows.append(
            {
                "vector_row_index": idx,
                "chunk_id": str(chunk_id),
                "embedding_model": manifest.get("embedding_model"),
                "embedding_dim": int(manifest.get("embedding_dim", 0)),
                "index_name": index_name,
                "indexed_at": now,
            }
        )
    return rows, manifest


def build_chunk_concept_rows(graph_dir: Path) -> List[Dict[str, str]]:
    concepts_path = graph_dir / "concepts.jsonl"
    if not concepts_path.exists():
        raise FileNotFoundError(f"Missing concepts file: {concepts_path}")

    concepts = read_jsonl(concepts_path)
    rows: List[Dict[str, str]] = []
    for c in concepts:
        cid = str(c["concept_id"])
        for chunk_id in c.get("mention_chunk_ids", []):
            rows.append({"chunk_id": str(chunk_id), "concept_id": cid})
    return rows


def write_sqlite_bindings(vector_dir: Path, mapping_rows: List[Dict[str, Any]], concept_rows: List[Dict[str, str]]) -> None:
    db_path = vector_dir / "metadata.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Vector metadata DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_index_map (
                vector_row_index INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_dim INTEGER NOT NULL,
                index_name TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_concept_map (
                chunk_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                PRIMARY KEY (chunk_id, concept_id)
            )
            """
        )

        conn.executemany(
            """
            INSERT INTO vector_index_map (
                vector_row_index, chunk_id, embedding_model, embedding_dim, index_name, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(vector_row_index) DO UPDATE SET
                chunk_id=excluded.chunk_id,
                embedding_model=excluded.embedding_model,
                embedding_dim=excluded.embedding_dim,
                index_name=excluded.index_name,
                indexed_at=excluded.indexed_at
            """,
            [
                (
                    r["vector_row_index"],
                    r["chunk_id"],
                    r["embedding_model"],
                    r["embedding_dim"],
                    r["index_name"],
                    r["indexed_at"],
                )
                for r in mapping_rows
            ],
        )

        conn.execute("DELETE FROM chunk_concept_map")
        conn.executemany(
            "INSERT INTO chunk_concept_map (chunk_id, concept_id) VALUES (?, ?)",
            [(r["chunk_id"], r["concept_id"]) for r in concept_rows],
        )

        conn.commit()
    finally:
        conn.close()


def write_neo4j_bindings(args: argparse.Namespace, mapping_rows: List[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    if args.skip_neo4j:
        return
    if not args.neo4j_password:
        raise ValueError("Neo4j password missing. Provide --neo4j-password or set NEO4J_PASSWORD.")

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_database) as session:
            session.run(
                """
                MERGE (v:VectorIndex {index_name: $index_name})
                SET v.embedding_model = $embedding_model,
                    v.embedding_dim = $embedding_dim,
                    v.chunks_path = $chunks_path,
                    v.updated_at = $updated_at
                """,
                {
                    "index_name": args.index_name,
                    "embedding_model": manifest.get("embedding_model"),
                    "embedding_dim": int(manifest.get("embedding_dim", 0)),
                    "chunks_path": manifest.get("chunks_path"),
                    "updated_at": utc_now_iso(),
                },
            ).consume()

            query = """
            UNWIND $rows AS row
            MATCH (c:Chunk {chunk_id: row.chunk_id})
            MATCH (v:VectorIndex {index_name: $index_name})
            MERGE (c)-[r:IN_VECTOR_INDEX]->(v)
            SET r.vector_row_index = row.vector_row_index,
                c.vector_row_index = row.vector_row_index,
                c.vector_index_name = $index_name,
                c.embedding_model = row.embedding_model,
                c.embedding_dim = row.embedding_dim
            """
            for batch in batched(mapping_rows, args.batch_size):
                session.run(query, {"rows": batch, "index_name": args.index_name}).consume()

            stats = session.run(
                """
                MATCH (:Chunk)-[r:IN_VECTOR_INDEX]->(:VectorIndex {index_name: $index_name})
                RETURN count(r) AS linked
                """,
                {"index_name": args.index_name},
            ).single()

        print(f"Neo4j vector bindings written. Linked chunks: {stats['linked']}")
    finally:
        driver.close()


def run() -> None:
    args = parse_args()

    mapping_rows, manifest = build_mapping_rows(args.vector_dir, args.index_name)
    concept_rows = build_chunk_concept_rows(args.graph_dir)

    write_sqlite_bindings(args.vector_dir, mapping_rows, concept_rows)
    write_neo4j_bindings(args, mapping_rows, manifest)

    print("Vector-Graph link complete.")
    print(f"Vector rows: {len(mapping_rows)}")
    print(f"Chunk-Concept rows: {len(concept_rows)}")
    print(f"SQLite DB: {args.vector_dir / 'metadata.sqlite'}")


if __name__ == "__main__":
    run()
