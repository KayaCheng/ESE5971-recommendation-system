#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.common import read_jsonl, utc_now_iso, write_jsonl


ALLOWED_RELATIONS = {
    "is_a",
    "part_of",
    "used_for",
    "depends_on",
    "compares_with",
    "measures",
    "produces",
    "implemented_by",
}

ALLOWED_TYPES = {
    "modality",
    "concept",
    "method",
    "signal",
    "artifact",
    "task",
    "metric",
    "anatomy",
    "physics_principle",
}


@dataclass
class ChunkRecord:
    chunk_id: str
    source_name: str
    page_start: int
    page_end: int
    chunk_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract concept graph candidates from chunk JSONL.")
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks/mis1to66_chunks.jsonl"),
        help="Path to chunk JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("storage/graph"),
        help="Directory for concept/relation outputs.",
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "heuristic"],
        default="heuristic",
        help="Extraction backend.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1-mini",
        help="Model name when backend=openai.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--max-chars", type=int, default=4000)
    return parser.parse_args()


def load_chunks(path: Path, max_chars: int) -> List[ChunkRecord]:
    raw = read_jsonl(path)
    chunks: List[ChunkRecord] = []
    for idx, item in enumerate(raw, start=1):
        missing = [k for k in ("chunk_id", "source_name", "page_start", "page_end", "chunk_text") if k not in item]
        if missing:
            raise ValueError(f"Record #{idx} missing required keys: {missing}")

        text = str(item["chunk_text"])
        if len(text) > max_chars:
            text = text[:max_chars]

        chunks.append(
            ChunkRecord(
                chunk_id=str(item["chunk_id"]),
                source_name=str(item["source_name"]),
                page_start=int(item["page_start"]),
                page_end=int(item["page_end"]),
                chunk_text=text,
            )
        )
    return chunks


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def concept_id(name: str, concept_type: str) -> str:
    return f"{slugify(concept_type)}::{slugify(name)}"


def parse_possible_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("Could not parse model output as JSON.")


def build_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is not installed.") from exc
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_with_openai(client: Any, model: str, chunk: ChunkRecord) -> Dict[str, Any]:
    schema_description = {
        "concepts": [
            {
                "name": "string",
                "type": f"one of {sorted(ALLOWED_TYPES)}",
                "aliases": ["string"],
                "description": "short string",
                "confidence": "float in [0,1]",
            }
        ],
        "relations": [
            {
                "source": "concept name string",
                "relation": f"one of {sorted(ALLOWED_RELATIONS)}",
                "target": "concept name string",
                "evidence": "short quote/paraphrase",
                "confidence": "float in [0,1]",
            }
        ],
    }

    system = (
        "You extract knowledge-graph candidates from medical imaging text. "
        "Output strict JSON only, no markdown. "
        "Do not invent concepts not grounded in the chunk."
    )

    user = (
        f"Chunk metadata: chunk_id={chunk.chunk_id}, source={chunk.source_name}, pages={chunk.page_start}-{chunk.page_end}.\n"
        f"Return object following this schema: {json.dumps(schema_description)}\n"
        f"Text:\n{chunk.chunk_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    content = response.choices[0].message.content or "{}"
    return parse_possible_json(content)


def extract_with_heuristic(chunk: ChunkRecord) -> Dict[str, Any]:
    text = chunk.chunk_text
    lower = text.lower()

    keyword_types = {
        "mri": "modality",
        "magnetic resonance imaging": "modality",
        "ct": "modality",
        "computed tomography": "modality",
        "x-ray": "modality",
        "ultrasound": "modality",
        "oct": "modality",
        "fourier transform": "method",
        "convolution": "method",
        "filter": "method",
        "signal": "signal",
        "image": "concept",
    }

    concepts = []
    seen = set()
    for key, ctype in keyword_types.items():
        if key in lower:
            name = key.upper() if len(key) <= 3 else key.title()
            cid = concept_id(name, ctype)
            if cid in seen:
                continue
            seen.add(cid)
            concepts.append(
                {
                    "name": name,
                    "type": ctype,
                    "aliases": [key],
                    "description": f"Keyword match for '{key}'.",
                    "confidence": 0.7,
                }
            )

    relations = []
    modality_names = [c["name"] for c in concepts if c["type"] == "modality"]
    for m in modality_names:
        relations.append(
            {
                "source": m,
                "relation": "used_for",
                "target": "Medical Imaging",
                "evidence": "Heuristic default relation.",
                "confidence": 0.6,
            }
        )

    if concepts and not any(c["name"] == "Medical Imaging" for c in concepts):
        concepts.append(
            {
                "name": "Medical Imaging",
                "type": "concept",
                "aliases": ["medical imaging"],
                "description": "Domain anchor concept.",
                "confidence": 0.8,
            }
        )

    return {"concepts": concepts, "relations": relations}


def normalize_concepts(raw_concepts: List[Dict[str, Any]], chunk: ChunkRecord, min_confidence: float) -> List[Dict[str, Any]]:
    normalized = []
    for c in raw_concepts:
        name = str(c.get("name", "")).strip()
        ctype = str(c.get("type", "concept")).strip().lower()
        if not name:
            continue
        if ctype not in ALLOWED_TYPES:
            ctype = "concept"

        confidence = float(c.get("confidence", 0.0))
        if confidence < min_confidence:
            continue

        aliases = c.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        normalized.append(
            {
                "concept_id": concept_id(name, ctype),
                "name": name,
                "type": ctype,
                "aliases": sorted({str(a).strip() for a in aliases if str(a).strip()}),
                "description": str(c.get("description", "")).strip(),
                "confidence": confidence,
                "chunk_id": chunk.chunk_id,
                "source_name": chunk.source_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
        )
    return normalized


def normalize_relations(
    raw_relations: List[Dict[str, Any]],
    concepts_for_chunk: List[Dict[str, Any]],
    chunk: ChunkRecord,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    by_name: Dict[str, str] = {}
    for c in concepts_for_chunk:
        by_name[c["name"].strip().lower()] = c["concept_id"]

    normalized = []
    for r in raw_relations:
        rel = str(r.get("relation", "")).strip().lower()
        if rel not in ALLOWED_RELATIONS:
            continue

        confidence = float(r.get("confidence", 0.0))
        if confidence < min_confidence:
            continue

        src_name = str(r.get("source", "")).strip()
        tgt_name = str(r.get("target", "")).strip()
        if not src_name or not tgt_name:
            continue

        src_id = by_name.get(src_name.lower())
        tgt_id = by_name.get(tgt_name.lower())

        if src_id is None or tgt_id is None:
            # Keep only fully grounded relations for now.
            continue

        normalized.append(
            {
                "relation_id": f"{src_id}|{rel}|{tgt_id}|{chunk.chunk_id}",
                "source_concept_id": src_id,
                "relation": rel,
                "target_concept_id": tgt_id,
                "confidence": confidence,
                "evidence": str(r.get("evidence", "")).strip(),
                "chunk_id": chunk.chunk_id,
                "source_name": chunk.source_name,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
            }
        )

    return normalized


def aggregate_concepts(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for c in concepts:
        cid = c["concept_id"]
        if cid not in grouped:
            grouped[cid] = {
                "concept_id": cid,
                "name": c["name"],
                "type": c["type"],
                "aliases": set(c.get("aliases", [])),
                "description": c.get("description", ""),
                "max_confidence": float(c["confidence"]),
                "mentions": set([c["chunk_id"]]),
            }
        else:
            grouped[cid]["aliases"].update(c.get("aliases", []))
            grouped[cid]["max_confidence"] = max(grouped[cid]["max_confidence"], float(c["confidence"]))
            grouped[cid]["mentions"].add(c["chunk_id"])
            if not grouped[cid]["description"] and c.get("description"):
                grouped[cid]["description"] = c.get("description", "")

    out = []
    for item in grouped.values():
        out.append(
            {
                "concept_id": item["concept_id"],
                "name": item["name"],
                "type": item["type"],
                "aliases": sorted(item["aliases"]),
                "description": item["description"],
                "max_confidence": item["max_confidence"],
                "mention_chunk_ids": sorted(item["mentions"]),
            }
        )
    out.sort(key=lambda x: x["concept_id"])
    return out


def dedup_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for r in relations:
        key = (
            r["source_concept_id"],
            r["relation"],
            r["target_concept_id"],
            r["chunk_id"],
        )
        old = best.get(key)
        if old is None or float(r["confidence"]) > float(old["confidence"]):
            best[key] = r
    out = list(best.values())
    out.sort(key=lambda x: x["relation_id"])
    return out


def run() -> None:
    args = parse_args()

    if not args.chunks_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {args.chunks_path}")

    chunks = load_chunks(args.chunks_path, max_chars=args.max_chars)
    if not chunks:
        raise ValueError("No chunks found in input JSONL.")

    client = None
    if args.backend == "openai":
        client = build_openai_client()

    concept_mentions: List[Dict[str, Any]] = []
    relation_mentions: List[Dict[str, Any]] = []
    logs: List[Dict[str, Any]] = []

    for chunk in chunks:
        try:
            if args.backend == "openai":
                result = extract_with_openai(client=client, model=args.llm_model, chunk=chunk)
            else:
                result = extract_with_heuristic(chunk)

            raw_concepts = result.get("concepts", [])
            raw_relations = result.get("relations", [])
            if not isinstance(raw_concepts, list) or not isinstance(raw_relations, list):
                raise ValueError("Model output must contain list fields: concepts, relations.")

            normalized_concepts = normalize_concepts(
                raw_concepts=raw_concepts,
                chunk=chunk,
                min_confidence=args.min_confidence,
            )
            normalized_relations = normalize_relations(
                raw_relations=raw_relations,
                concepts_for_chunk=normalized_concepts,
                chunk=chunk,
                min_confidence=args.min_confidence,
            )

            concept_mentions.extend(normalized_concepts)
            relation_mentions.extend(normalized_relations)

            logs.append(
                {
                    "timestamp": utc_now_iso(),
                    "chunk_id": chunk.chunk_id,
                    "status": "ok",
                    "concept_count": len(normalized_concepts),
                    "relation_count": len(normalized_relations),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logs.append(
                {
                    "timestamp": utc_now_iso(),
                    "chunk_id": chunk.chunk_id,
                    "status": "error",
                    "error": str(exc),
                }
            )

    concepts = aggregate_concepts(concept_mentions)
    relations = dedup_relations(relation_mentions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(concepts, args.output_dir / "concepts.jsonl")
    write_jsonl(relations, args.output_dir / "relations.jsonl")
    write_jsonl(logs, args.output_dir / "extraction_log.jsonl")

    summary = {
        "created_at": utc_now_iso(),
        "backend": args.backend,
        "llm_model": args.llm_model if args.backend == "openai" else None,
        "chunks_processed": len(chunks),
        "concepts_total": len(concepts),
        "relations_total": len(relations),
        "log_errors": sum(1 for x in logs if x.get("status") == "error"),
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Concept extraction complete.")
    print(f"Backend: {args.backend}")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Concepts: {len(concepts)}")
    print(f"Relations: {len(relations)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    run()
