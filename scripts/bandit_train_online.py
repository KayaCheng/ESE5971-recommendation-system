#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.bandit import LinUCB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/update LinUCB model from bandit event log.")
    parser.add_argument("--events-path", type=Path, default=Path("storage/bandit/events.jsonl"))
    parser.add_argument("--model-path", type=Path, default=Path("storage/bandit/model_linucb.json"))
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--l2-lambda", type=float, default=1.0)
    parser.add_argument("--reset", action="store_true", help="Ignore existing model and retrain from scratch.")
    return parser.parse_args()


def iter_events(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def run() -> None:
    args = parse_args()
    if not args.events_path.exists():
        raise FileNotFoundError(f"Events file not found: {args.events_path}")

    model = None
    seen = 0
    skipped = 0
    reward_sum = 0.0

    for event in iter_events(args.events_path):
        chosen = event.get("chosen_arm", {})
        feat = chosen.get("feature")
        reward = event.get("reward")
        if feat is None or reward is None:
            skipped += 1
            continue

        x = np.asarray(feat, dtype=np.float64)
        if model is None:
            if args.model_path.exists() and not args.reset:
                loaded = LinUCB.load(args.model_path)
                if loaded.dim != x.shape[0]:
                    raise RuntimeError(
                        f"Model dim mismatch: model={loaded.dim}, event={x.shape[0]}. Use --reset or matching logs."
                    )
                model = loaded
            else:
                model = LinUCB(dim=x.shape[0], alpha=args.alpha, l2_lambda=args.l2_lambda)

        model.update(x, float(reward))
        seen += 1
        reward_sum += float(reward)

    if model is None:
        raise RuntimeError("No trainable events found.")

    model.save(args.model_path)

    print("Bandit online training complete.")
    print(f"Events used: {seen}")
    print(f"Events skipped: {skipped}")
    print(f"Average reward in training stream: {reward_sum / max(1, seen):.4f}")
    print(f"Model saved to: {args.model_path}")


if __name__ == "__main__":
    run()
