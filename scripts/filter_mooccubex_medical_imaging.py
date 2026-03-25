#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.mooccubex_utils import (
    any_keyword_hit,
    find_files,
    infer_id,
    infer_src_dst,
    infer_text,
    infer_user_item_time,
    iter_records,
    write_jsonl,
)


DEFAULT_KEYWORDS = [
    "medical imaging",
    "radiology",
    "x-ray",
    "mri",
    "ct",
    "computed tomography",
    "magnetic resonance",
    "ultrasound",
    "pet",
    "spect",
    "image reconstruction",
    "segmentation",
    "diagnostic imaging",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter MOOCCubeX for medical imaging topic.")
    parser.add_argument("--mooc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("storage/mooccubex_medimg"))
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--keywords-file", type=Path, default=None)
    parser.add_argument("--max-events", type=int, default=200000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_keywords(args: argparse.Namespace) -> List[str]:
    kws = [k.lower().strip() for k in (args.keywords or []) if k.strip()]
    if args.keywords_file and args.keywords_file.exists():
        lines = [x.strip().lower() for x in args.keywords_file.read_text(encoding="utf-8").splitlines()]
        kws.extend([x for x in lines if x])
    if not kws:
        kws = list(DEFAULT_KEYWORDS)
    return sorted(set(kws))


def _subroot(root: Path, name: str) -> Path:
    p = root / name
    return p if p.exists() and p.is_dir() else root


def collect_entity_files(root: Path) -> Dict[str, List[Path]]:
    base = _subroot(root, "entities")
    return {
        "concept": find_files(base, ["concept"]),
        "video": find_files(base, ["video"]),
        "course": find_files(base, ["course"]),
        "problem": find_files(base, ["problem", "exercise"]),
    }


def collect_relation_files(root: Path) -> Dict[str, List[Path]]:
    base = _subroot(root, "relations")
    return {
        "concept_video": find_files(base, ["concept-video", "concept_video", "video-concept"]),
        "concept_course": find_files(base, ["concept-course", "concept_course", "course-concept"]),
        "concept_problem": find_files(base, ["concept-problem", "concept_problem", "problem-concept"]),
        "user_video": find_files(base, ["user-video", "user_video", "video-watch", "watch"]),
        "user_course": find_files(base, ["user-course", "user_course", "course-learn"]),
        "user_problem": find_files(base, ["user-problem", "user_problem", "problem-submit", "submit"]),
    }


def normalize_entities(files: List[Path], kind: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in files:
        for row in iter_records(path):
            eid = infer_id(row)
            if not eid:
                continue
            text = infer_text(row)
            prev = out.get(eid)
            if prev is None or (len(text) > len(prev.get("text", ""))):
                out[eid] = {
                    "id": eid,
                    "kind": kind,
                    "text": text,
                    "source_file": str(path),
                }
    return out


def mark_topic_entities(entities: Dict[str, Dict[str, Any]], keywords: List[str]) -> Set[str]:
    selected: Set[str] = set()
    for eid, row in entities.items():
        if any_keyword_hit(row.get("text", ""), keywords):
            selected.add(eid)
    return selected


def expand_by_relations(
    rel_files: List[Path],
    left_selected: Set[str],
    right_selected: Set[str],
    two_way: bool = True,
) -> tuple[Set[str], Set[str]]:
    left = set(left_selected)
    right = set(right_selected)

    changed = True
    while changed:
        changed = False
        for path in rel_files:
            for row in iter_records(path):
                a, b = infer_src_dst(row)
                if not a or not b:
                    continue
                if a in left and b not in right:
                    right.add(b)
                    changed = True
                if two_way and b in right and a not in left:
                    left.add(a)
                    changed = True
    return left, right


def gather_user_events(rel_files: List[Path], allowed_items: Set[str], event_type: str, max_events: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen = 0
    for path in rel_files:
        for row in iter_records(path):
            user_id, item_id, ts = infer_user_item_time(row, seen)
            if not user_id or not item_id:
                continue
            if item_id not in allowed_items:
                continue
            events.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event_type": event_type,
                    "timestamp": ts,
                    "source_file": str(path),
                }
            )
            seen += 1
            if seen >= max_events:
                return events
    return events


def run() -> None:
    args = parse_args()
    if not args.mooc_root.exists():
        raise FileNotFoundError(f"MOOCCubeX root not found: {args.mooc_root}")

    keywords = load_keywords(args)

    entity_files = collect_entity_files(args.mooc_root)
    rel_files = collect_relation_files(args.mooc_root)

    concepts = normalize_entities(entity_files["concept"], "concept")
    videos = normalize_entities(entity_files["video"], "video")
    courses = normalize_entities(entity_files["course"], "course")
    problems = normalize_entities(entity_files["problem"], "problem")

    selected_concepts = mark_topic_entities(concepts, keywords)
    selected_videos = mark_topic_entities(videos, keywords)
    selected_courses = mark_topic_entities(courses, keywords)
    selected_problems = mark_topic_entities(problems, keywords)

    # Expand selections via concept relations
    if rel_files["concept_video"]:
        selected_concepts, selected_videos = expand_by_relations(
            rel_files["concept_video"], selected_concepts, selected_videos
        )
    if rel_files["concept_course"]:
        selected_concepts, selected_courses = expand_by_relations(
            rel_files["concept_course"], selected_concepts, selected_courses
        )
    if rel_files["concept_problem"]:
        selected_concepts, selected_problems = expand_by_relations(
            rel_files["concept_problem"], selected_concepts, selected_problems
        )

    allowed_items = set(selected_videos) | set(selected_courses) | set(selected_problems)

    events = []
    events.extend(gather_user_events(rel_files["user_video"], selected_videos, "watch_video", args.max_events))
    events.extend(gather_user_events(rel_files["user_course"], selected_courses, "learn_course", args.max_events))
    events.extend(gather_user_events(rel_files["user_problem"], selected_problems, "solve_problem", args.max_events))

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl((concepts[c] for c in sorted(selected_concepts) if c in concepts), args.output_dir / "concepts_filtered.jsonl")
    write_jsonl((videos[v] for v in sorted(selected_videos) if v in videos), args.output_dir / "videos_filtered.jsonl")
    write_jsonl((courses[c] for c in sorted(selected_courses) if c in courses), args.output_dir / "courses_filtered.jsonl")
    write_jsonl((problems[p] for p in sorted(selected_problems) if p in problems), args.output_dir / "problems_filtered.jsonl")
    write_jsonl(events, args.output_dir / "events_filtered.jsonl")

    summary = {
        "keywords": keywords,
        "entity_files": {k: [str(x) for x in v] for k, v in entity_files.items()},
        "relation_files": {k: [str(x) for x in v] for k, v in rel_files.items()},
        "selected_counts": {
            "concepts": len(selected_concepts),
            "videos": len(selected_videos),
            "courses": len(selected_courses),
            "problems": len(selected_problems),
            "items_total": len(allowed_items),
            "events": len(events),
        },
        "output_dir": str(args.output_dir),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("MOOCCubeX topic filtering complete.")
    print(f"Concepts: {len(selected_concepts)}")
    print(f"Videos: {len(selected_videos)}")
    print(f"Courses: {len(selected_courses)}")
    print(f"Problems: {len(selected_problems)}")
    print(f"Events: {len(events)}")
    print(f"Output: {args.output_dir}")

    if args.verbose:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
