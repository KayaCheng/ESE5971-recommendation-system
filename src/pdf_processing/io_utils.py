'''
1. save jsonl
2. read jsonl
'''

from pathlib import Path
import json


def save_jsonl(records, output_path: Path):
    """
    Save a list of dict records to JSONL.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")