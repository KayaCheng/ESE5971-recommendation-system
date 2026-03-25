#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.mooccubex_utils import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contextual bandit events from filtered MOOCCubeX data.")
    parser.add_argument("--filtered-dir", type=Path, default=Path("storage/mooccubex_medimg"))
    parser.add_argument("--events-out", type=Path, default=Path("storage/mooccubex_medimg/bandit_events.jsonl"))
    parser.add_argument("--top-k", type=int, default=10, help="Candidate set size per decision.")
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--min-user-events", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def lexical_score(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    return inter / max(1, union)


def load_items(filtered_dir: Path) -> Dict[str, Dict[str, Any]]:
    items: Dict[str, Dict[str, Any]] = {}
    for name in ["videos_filtered.jsonl", "courses_filtered.jsonl", "problems_filtered.jsonl"]:
        p = filtered_dir / name
        if not p.exists():
            continue
        for row in read_jsonl(p):
            item_id = str(row["id"])
            items[item_id] = {
                "item_id": item_id,
                "kind": row.get("kind", "item"),
                "text": row.get("text", ""),
            }
    return items


def make_user_state(history: Deque[Dict[str, Any]], items: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {
            "history_len": 0,
            "unique_items": 0,
            "recent_event_type_ratio_watch": 0.0,
            "recent_event_type_ratio_problem": 0.0,
            "recent_text": "",
        }

    item_ids = [h["item_id"] for h in history]
    unique_items = len(set(item_ids))
    watch_cnt = sum(1 for h in history if "video" in h.get("event_type", ""))
    prob_cnt = sum(1 for h in history if "problem" in h.get("event_type", ""))

    recent_text = " ".join(items.get(i, {}).get("text", "") for i in item_ids[-5:])

    return {
        "history_len": len(history),
        "unique_items": unique_items,
        "recent_event_type_ratio_watch": watch_cnt / max(1, len(history)),
        "recent_event_type_ratio_problem": prob_cnt / max(1, len(history)),
        "recent_text": recent_text,
    }


def make_feature(user_state: Dict[str, Any], candidate_text: str, candidate_pop: float) -> List[float]:
    overlap = lexical_score(user_state.get("recent_text", ""), candidate_text)
    hist_len_norm = min(1.0, user_state.get("history_len", 0) / 50.0)
    unique_norm = min(1.0, user_state.get("unique_items", 0) / 50.0)
    watch_ratio = float(user_state.get("recent_event_type_ratio_watch", 0.0))
    problem_ratio = float(user_state.get("recent_event_type_ratio_problem", 0.0))
    pop_norm = min(1.0, candidate_pop)
    return [hist_len_norm, unique_norm, watch_ratio, problem_ratio, overlap, pop_norm]


def simulate_reward(chosen_item_text: str, next_item_text: str) -> float:
    # proxy for learning continuity / topical coherence
    sim = lexical_score(chosen_item_text, next_item_text)
    return max(0.0, min(1.0, 0.2 + 0.8 * sim))


def run() -> None:
    args = parse_args()
    random.seed(args.seed)

    events_path = args.filtered_dir / "events_filtered.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Filtered events not found: {events_path}")

    items = load_items(args.filtered_dir)
    if not items:
        raise RuntimeError("No filtered items found. Run topic filtering first.")

    raw_events = read_jsonl(events_path)
    if not raw_events:
        raise RuntimeError("No filtered user events found.")

    # sort by user, timestamp (string order fallback)
    raw_events.sort(key=lambda x: (str(x.get("user_id", "")), str(x.get("timestamp", ""))))

    item_pop = defaultdict(int)
    for ev in raw_events:
        iid = str(ev.get("item_id", ""))
        if iid in items:
            item_pop[iid] += 1
    max_pop = max(item_pop.values()) if item_pop else 1

    per_user_history: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=args.history_size))
    args.events_out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.events_out.open("w", encoding="utf-8") as f:
        for idx, ev in enumerate(raw_events):
            user_id = str(ev.get("user_id", ""))
            item_id = str(ev.get("item_id", ""))
            if not user_id or item_id not in items:
                continue

            history = per_user_history[user_id]

            # Need at least min history before creating decision event.
            if len(history) >= args.min_user_events:
                # candidate pool: true item + negatives
                pool = [iid for iid in items.keys() if iid != item_id]
                neg_num = max(0, args.top_k - 1)
                negatives = random.sample(pool, k=min(neg_num, len(pool))) if pool else []
                candidate_ids = [item_id] + negatives
                random.shuffle(candidate_ids)

                user_state = make_user_state(history, items)

                candidate_arms = []
                chosen_index = None
                for arm_idx, cid in enumerate(candidate_ids):
                    ctext = items[cid]["text"]
                    pop_norm = item_pop.get(cid, 0) / max_pop
                    feat = make_feature(user_state, ctext, pop_norm)
                    candidate_arms.append(
                        {
                            "arm_index": arm_idx,
                            "chunk_id": cid,
                            "item_kind": items[cid].get("kind"),
                            "feature": feat,
                            "retrieval_score": feat[-2],  # lexical overlap proxy
                        }
                    )
                    if cid == item_id:
                        chosen_index = arm_idx

                if chosen_index is None:
                    continue

                # logged policy is uniform over shuffled candidate set for seed dataset
                propensity = 1.0 / max(1, len(candidate_arms))

                # reward proxy: coherence between chosen item and next event of same user (if exists)
                next_text = ""
                if idx + 1 < len(raw_events) and str(raw_events[idx + 1].get("user_id", "")) == user_id:
                    next_id = str(raw_events[idx + 1].get("item_id", ""))
                    next_text = items.get(next_id, {}).get("text", "")
                reward = simulate_reward(items[item_id]["text"], next_text)

                out = {
                    "event_id": str(uuid.uuid4()),
                    "user_id": user_id,
                    "ts": str(ev.get("timestamp", f"order_{idx:09d}")),
                    "policy_name": "logged_uniform_candidates",
                    "candidate_arms": candidate_arms,
                    "chosen_arm": {
                        "arm_index": chosen_index,
                        "chunk_id": item_id,
                        "feature": candidate_arms[chosen_index]["feature"],
                        "propensity": propensity,
                    },
                    "reward": reward,
                    "user_state": {
                        k: v for k, v in user_state.items() if k != "recent_text"
                    },
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1

            history.append(ev)

    summary = {
        "filtered_dir": str(args.filtered_dir),
        "items": len(items),
        "raw_events": len(raw_events),
        "bandit_events": written,
        "feature_dim": 6,
        "events_out": str(args.events_out),
    }
    with (args.events_out.parent / "bandit_events_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Bandit event building complete.")
    print(f"Items: {len(items)}")
    print(f"Raw events: {len(raw_events)}")
    print(f"Bandit events: {written}")
    print(f"Output: {args.events_out}")


if __name__ == "__main__":
    run()
