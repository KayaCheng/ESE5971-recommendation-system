from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Sequence


ID_KEYS = [
    "id",
    "_id",
    "course_id",
    "video_id",
    "problem_id",
    "concept_id",
    "entity_id",
    "uid",
    "user_id",
]

TEXT_KEYS = [
    "name",
    "title",
    "text",
    "content",
    "description",
    "desc",
    "caption",
    "keywords",
    "summary",
]

TIME_KEYS = [
    "timestamp",
    "time",
    "ts",
    "date",
    "event_time",
    "watch_time",
    "submit_time",
]

USER_KEYS = ["user_id", "uid", "user", "student_id"]
ITEM_KEYS = [
    "video_id",
    "course_id",
    "problem_id",
    "resource_id",
    "item_id",
    "target_id",
    "object_id",
    "to_id",
]
SRC_KEYS = ["source", "src", "from", "from_id", "head", "concept_id"]
DST_KEYS = ["target", "dst", "to", "to_id", "tail", "video_id", "course_id", "problem_id"]


def normalize_space(text: str) -> str:
    return " ".join(str(text).split())


def safe_lower(text: str) -> str:
    return normalize_space(text).lower()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", safe_lower(text))


def any_keyword_hit(text: str, keywords: Sequence[str]) -> bool:
    t = safe_lower(text)
    return any(k in t for k in keywords)


def pick_first(record: Dict[str, Any], candidates: Sequence[str]) -> Any:
    low_map = {k.lower(): v for k, v in record.items()}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


def infer_id(record: Dict[str, Any]) -> str:
    val = pick_first(record, ID_KEYS)
    if val is None:
        # fallback: first scalar field
        for _, v in record.items():
            if isinstance(v, (str, int, float)):
                return str(v)
        return ""
    return str(val)


def infer_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    low_map = {k.lower(): v for k, v in record.items()}
    for key in TEXT_KEYS:
        if key in low_map and low_map[key] is not None:
            parts.append(str(low_map[key]))
    if not parts:
        for k, v in record.items():
            if isinstance(v, str) and len(v) > 20:
                parts.append(v)
    return normalize_space(" ".join(parts))


def infer_time(record: Dict[str, Any], default_order: int = 0) -> str:
    val = pick_first(record, TIME_KEYS)
    if val is None:
        return f"order_{default_order:09d}"
    return str(val)


def parse_json_or_jsonl(path: Path) -> Generator[Dict[str, Any], None, None]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return

    # JSON array/object
    if text.startswith("[") or text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            for row in payload:
                if isinstance(row, dict):
                    yield row
                else:
                    yield {"value": row}
            return
        if isinstance(payload, dict):
            # if dict of dicts
            if all(isinstance(v, dict) for v in payload.values()):
                for v in payload.values():
                    yield v
            else:
                yield payload
            return

    # JSONL fallback
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
                else:
                    yield {"value": row}
            except json.JSONDecodeError:
                continue


def parse_delimited(path: Path) -> Generator[Dict[str, Any], None, None]:
    sample = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not sample:
        return

    first = sample[0]
    delim = "\t" if "\t" in first else ","

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    if not rows:
        return

    header = rows[0]
    looks_header = any(re.search(r"[a-zA-Z_]", c or "") for c in header)

    start = 1 if looks_header else 0
    if looks_header:
        columns = [normalize_space(c) if c else f"col_{i}" for i, c in enumerate(header)]
    else:
        n = max(len(r) for r in rows)
        columns = [f"col_{i}" for i in range(n)]

    for row in rows[start:]:
        if not row:
            continue
        record = {columns[i]: row[i] if i < len(row) else "" for i in range(len(columns))}
        yield record


def iter_records(path: Path) -> Generator[Dict[str, Any], None, None]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        yield from parse_json_or_jsonl(path)
    elif suffix in {".txt", ".tsv", ".csv"}:
        yield from parse_delimited(path)
    else:
        # last chance: try json/jsonl then delimited
        yielded = False
        for row in parse_json_or_jsonl(path):
            yielded = True
            yield row
        if not yielded:
            yield from parse_delimited(path)


def find_files(root: Path, name_hints: Sequence[str]) -> List[Path]:
    matched: List[Path] = []
    hints = [h.lower() for h in name_hints]
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".jsonl", ".txt", ".tsv", ".csv"}:
            continue
        name = p.name.lower()
        if any(h in name for h in hints):
            matched.append(p)
    return sorted(matched)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def infer_user_item_time(record: Dict[str, Any], row_idx: int = 0) -> tuple[str, str, str]:
    user = pick_first(record, USER_KEYS)
    item = pick_first(record, ITEM_KEYS)

    if user is None or item is None:
        # fallback by column order if relation row
        vals = [str(v) for v in record.values()]
        if len(vals) >= 2:
            user = vals[0] if user is None else user
            item = vals[1] if item is None else item

    if user is None or item is None:
        return "", "", infer_time(record, row_idx)

    return str(user), str(item), infer_time(record, row_idx)


def infer_src_dst(record: Dict[str, Any]) -> tuple[str, str]:
    src = pick_first(record, SRC_KEYS)
    dst = pick_first(record, DST_KEYS)

    if src is None or dst is None:
        vals = [str(v) for v in record.values()]
        if len(vals) >= 2:
            src = vals[0] if src is None else src
            dst = vals[1] if dst is None else dst

    return ("" if src is None else str(src), "" if dst is None else str(dst))
