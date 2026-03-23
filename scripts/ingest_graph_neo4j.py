#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest chunk/concept/relation JSONL into Neo4j.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks/mis1to66_chunks.jsonl"),
    )
    parser.add_argument(
        "--concepts-path",
        type=Path,
        default=Path("storage/graph/concepts.jsonl"),
    )
    parser.add_argument(
        "--relations-path",
        type=Path,
        default=Path("storage/graph/relations.jsonl"),
    )
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--skip-constraints", action="store_true")
    return parser.parse_args()


def batched(rows: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def build_chunk_rows(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in chunks:
        rows.append(
            {
                "chunk_id": str(item["chunk_id"]),
                "source_name": str(item["source_name"]),
                "page_start": int(item["page_start"]),
                "page_end": int(item["page_end"]),
                "chunk_text": str(item.get("chunk_text", "")),
                "char_count": int(item.get("char_count", 0)),
                "token_estimate": int(item.get("token_estimate", 0)),
                "prev_chunk_id": item.get("prev_chunk_id"),
                "next_chunk_id": item.get("next_chunk_id"),
            }
        )
    return rows


def build_mentions(concepts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for c in concepts:
        cid = str(c["concept_id"])
        for chunk_id in c.get("mention_chunk_ids", []):
            rows.append({"chunk_id": str(chunk_id), "concept_id": cid})
    return rows


def build_next_edges(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for item in chunks:
        src = item.get("chunk_id")
        nxt = item.get("next_chunk_id")
        if src and nxt:
            rows.append({"from_chunk_id": str(src), "to_chunk_id": str(nxt)})
    return rows


def create_constraints(session) -> None:
    queries = [
        "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
        "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE",
        "CREATE CONSTRAINT vector_index_name_unique IF NOT EXISTS FOR (v:VectorIndex) REQUIRE v.index_name IS UNIQUE",
    ]
    for q in queries:
        session.run(q).consume()


def ingest_chunks(session, rows: List[Dict[str, Any]], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (c:Chunk {chunk_id: row.chunk_id})
    SET c.source_name = row.source_name,
        c.page_start = row.page_start,
        c.page_end = row.page_end,
        c.chunk_text = row.chunk_text,
        c.char_count = row.char_count,
        c.token_estimate = row.token_estimate,
        c.prev_chunk_id = row.prev_chunk_id,
        c.next_chunk_id = row.next_chunk_id
    """
    for batch in batched(rows, batch_size):
        session.run(query, {"rows": batch}).consume()


def ingest_concepts(session, rows: List[Dict[str, Any]], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MERGE (c:Concept {concept_id: row.concept_id})
    SET c.name = row.name,
        c.type = row.type,
        c.aliases = row.aliases,
        c.description = row.description,
        c.max_confidence = row.max_confidence
    """
    for batch in batched(rows, batch_size):
        session.run(query, {"rows": batch}).consume()


def ingest_mentions(session, rows: List[Dict[str, str]], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (ch:Chunk {chunk_id: row.chunk_id})
    MATCH (co:Concept {concept_id: row.concept_id})
    MERGE (ch)-[:MENTIONS]->(co)
    """
    for batch in batched(rows, batch_size):
        session.run(query, {"rows": batch}).consume()


def ingest_next_edges(session, rows: List[Dict[str, str]], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (a:Chunk {chunk_id: row.from_chunk_id})
    MATCH (b:Chunk {chunk_id: row.to_chunk_id})
    MERGE (a)-[:NEXT]->(b)
    """
    for batch in batched(rows, batch_size):
        session.run(query, {"rows": batch}).consume()


def ingest_relations(session, rows: List[Dict[str, Any]], batch_size: int) -> None:
    query = """
    UNWIND $rows AS row
    MATCH (s:Concept {concept_id: row.source_concept_id})
    MATCH (t:Concept {concept_id: row.target_concept_id})
    MERGE (s)-[r:RELATES {relation_id: row.relation_id}]->(t)
    SET r.type = row.relation,
        r.confidence = row.confidence,
        r.evidence = row.evidence,
        r.chunk_id = row.chunk_id,
        r.source_name = row.source_name,
        r.page_start = row.page_start,
        r.page_end = row.page_end
    """
    for batch in batched(rows, batch_size):
        session.run(query, {"rows": batch}).consume()


def run() -> None:
    args = parse_args()
    if not args.neo4j_password:
        raise ValueError("Neo4j password missing. Provide --neo4j-password or set NEO4J_PASSWORD.")

    for p in (args.chunks_path, args.concepts_path, args.relations_path):
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    chunks = read_jsonl(args.chunks_path)
    concepts = read_jsonl(args.concepts_path)
    relations = read_jsonl(args.relations_path)

    chunk_rows = build_chunk_rows(chunks)
    concept_rows = concepts
    mention_rows = build_mentions(concepts)
    next_rows = build_next_edges(chunks)

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    try:
        with driver.session(database=args.neo4j_database) as session:
            if not args.skip_constraints:
                create_constraints(session)

            ingest_chunks(session, chunk_rows, args.batch_size)
            ingest_concepts(session, concept_rows, args.batch_size)
            ingest_mentions(session, mention_rows, args.batch_size)
            ingest_next_edges(session, next_rows, args.batch_size)
            ingest_relations(session, relations, args.batch_size)

            stats = session.run(
                """
                MATCH (c:Chunk) WITH count(c) AS chunks
                MATCH (k:Concept) WITH chunks, count(k) AS concepts
                MATCH ()-[m:MENTIONS]->() WITH chunks, concepts, count(m) AS mentions
                MATCH ()-[r:RELATES]->() WITH chunks, concepts, mentions, count(r) AS relations
                MATCH ()-[n:NEXT]->() RETURN chunks, concepts, mentions, relations, count(n) AS next_edges
                """
            ).single()

        print("Neo4j ingestion complete.")
        print(f"Chunk nodes: {stats['chunks']}")
        print(f"Concept nodes: {stats['concepts']}")
        print(f"MENTIONS edges: {stats['mentions']}")
        print(f"RELATES edges: {stats['relations']}")
        print(f"NEXT edges: {stats['next_edges']}")
    finally:
        driver.close()


if __name__ == "__main__":
    run()
