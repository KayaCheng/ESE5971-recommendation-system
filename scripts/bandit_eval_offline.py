#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_pipeline.bandit import LinUCB, ridge_fit, ridge_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline OPE for baseline vs bandit policies.")
    parser.add_argument("--events-path", type=Path, default=Path("storage/bandit/events.jsonl"))
    parser.add_argument("--model-path", type=Path, default=Path("storage/bandit/model_linucb.json"))
    parser.add_argument("--target-epsilon", type=float, default=0.05)
    parser.add_argument("--reward-model-reg", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=Path("storage/bandit/offline_eval_report.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_events(path: Path) -> List[Dict[str, Any]]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def action_prob(best_idx: int, chosen_idx: int, n_arms: int, epsilon: float) -> float:
    if chosen_idx == best_idx:
        return (1.0 - epsilon) + epsilon / n_arms
    return epsilon / n_arms


def policy_best_index(policy: str, event: Dict[str, Any], model: LinUCB | None, rng: np.random.Generator) -> int:
    arms = event["candidate_arms"]
    if policy == "retrieval":
        scores = [float(a["retrieval_score"]) for a in arms]
        return int(np.argmax(scores))
    if policy == "random":
        return int(rng.integers(0, len(arms)))
    if policy == "linucb":
        if model is None:
            raise RuntimeError("LinUCB model required for linucb policy evaluation.")
        features = [np.asarray(a["feature"], dtype=np.float64) for a in arms]
        return model.choose_index(features)
    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policy(
    events: List[Dict[str, Any]],
    policy_name: str,
    target_epsilon: float,
    reward_w: np.ndarray,
    model: LinUCB | None,
    rng: np.random.Generator,
) -> Dict[str, float]:
    ips_num = 0.0
    ips_den = 0.0
    dr_sum = 0.0
    w_sum = 0.0
    direct_sum = 0.0

    for ev in events:
        arms = ev["candidate_arms"]
        K = len(arms)
        logged_idx = int(ev["chosen_arm"]["arm_index"])
        logged_p = float(ev["chosen_arm"]["propensity"])
        reward = float(ev["reward"])

        best_idx = policy_best_index(policy_name, ev, model, rng)
        pi_logged = action_prob(best_idx, logged_idx, K, target_epsilon)
        weight = pi_logged / max(1e-8, logged_p)

        # IPS/SNIPS
        ips_num += weight * reward
        ips_den += 1.0
        w_sum += weight

        # DR components
        X = np.asarray([a["feature"] for a in arms], dtype=np.float64)
        q_hat = ridge_predict(X, reward_w)  # predicted reward for each arm

        probs = np.full(K, target_epsilon / K, dtype=np.float64)
        probs[best_idx] += 1.0 - target_epsilon

        v_hat = float(np.sum(probs * q_hat))
        dr = v_hat + weight * (reward - float(q_hat[logged_idx]))
        dr_sum += dr
        direct_sum += v_hat

    n = max(1, len(events))
    ips = ips_num / ips_den
    snips = ips_num / max(1e-8, w_sum)
    dr = dr_sum / n
    dm = direct_sum / n

    return {
        "IPS": float(ips),
        "SNIPS": float(snips),
        "DR": float(dr),
        "DM": float(dm),
        "n_events": len(events),
    }


def bootstrap_ci(
    events: List[Dict[str, Any]],
    policy_name: str,
    target_epsilon: float,
    reward_w: np.ndarray,
    model: LinUCB | None,
    seed: int,
    n_boot: int = 200,
) -> Dict[str, List[float]]:
    rng = np.random.default_rng(seed)
    dr_values = []
    n = len(events)
    if n == 0:
        return {"DR_95CI": [0.0, 0.0]}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [events[int(i)] for i in idx]
        metric = evaluate_policy(
            events=sample,
            policy_name=policy_name,
            target_epsilon=target_epsilon,
            reward_w=reward_w,
            model=model,
            rng=rng,
        )
        dr_values.append(metric["DR"])

    lo = float(np.quantile(dr_values, 0.025))
    hi = float(np.quantile(dr_values, 0.975))
    return {"DR_95CI": [lo, hi]}


def run() -> None:
    args = parse_args()
    if not args.events_path.exists():
        raise FileNotFoundError(f"Events file not found: {args.events_path}")

    events = load_events(args.events_path)
    if not events:
        raise RuntimeError("No events to evaluate.")

    # Fit reward model q(x,a) with ridge on logged action features.
    X = np.asarray([ev["chosen_arm"]["feature"] for ev in events], dtype=np.float64)
    y = np.asarray([float(ev["reward"]) for ev in events], dtype=np.float64)
    w = ridge_fit(X, y, reg=args.reward_model_reg)

    linucb = None
    if args.model_path.exists():
        linucb = LinUCB.load(args.model_path)
        if linucb.dim != X.shape[1]:
            raise RuntimeError(
                f"Model dim mismatch: model={linucb.dim}, event_feature_dim={X.shape[1]}"
            )

    rng = np.random.default_rng(args.seed)

    report: Dict[str, Any] = {
        "events_path": str(args.events_path),
        "target_epsilon": args.target_epsilon,
        "reward_model_reg": args.reward_model_reg,
        "policies": {},
    }

    for policy in ("retrieval", "random", "linucb"):
        if policy == "linucb" and linucb is None:
            continue

        metrics = evaluate_policy(
            events=events,
            policy_name=policy,
            target_epsilon=args.target_epsilon,
            reward_w=w,
            model=linucb,
            rng=rng,
        )
        ci = bootstrap_ci(
            events=events,
            policy_name=policy,
            target_epsilon=args.target_epsilon,
            reward_w=w,
            model=linucb,
            seed=args.seed + 17,
        )
        report["policies"][policy] = {**metrics, **ci}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Offline bandit evaluation complete.")
    print(f"Events: {len(events)}")
    for name, m in report["policies"].items():
        print(
            f"{name}: DR={m['DR']:.4f} (95%CI [{m['DR_95CI'][0]:.4f}, {m['DR_95CI'][1]:.4f}]), "
            f"IPS={m['IPS']:.4f}, SNIPS={m['SNIPS']:.4f}"
        )
    print(f"Report: {args.output_json}")


if __name__ == "__main__":
    run()
