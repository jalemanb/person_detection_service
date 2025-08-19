#!/usr/bin/env python3
"""
Summarize timings from eval_results.log, dropping the first N occurrences
(per metric) IN-MEMORY, and printing results to the terminal.

Usage example:
  python3 log_stats.py \
    --file /path/to/eval_results.log \
    --drop-first 10 \
    --metrics detection tracking reid train
"""
import argparse
import re
import statistics
from pathlib import Path
from collections import defaultdict

# Match the message portion after " - INFO - "
KEYVAL = re.compile(r"-\s+INFO\s+-\s+(?P<msg>.*)")
# Match key: value pairs within the message (supports scientific notation)
PAIR = re.compile(r"(?P<key>[A-Za-z][\w-]*)\s*:\s*(?P<val>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

# Normalize variants/typos → canonical metric names
SYNONYMS = {
    "tracknig": "tracking",
    "id": "reid",
    "re-id": "reid",
    "kpr": "reid",
    "reid_time": "reid",
    "kpr_time": "reid",
}

def parse_args():
    ap = argparse.ArgumentParser(description="Compute mean/min/max/std in ms after dropping first N occurrences per metric.")
    ap.add_argument("--file", type=Path, required=True, help="Path to eval_results.log")
    ap.add_argument("--drop-first", type=int, default=10, help="Warm-up samples to drop per metric (default: 10)")
    ap.add_argument("--metrics", nargs="*", default=["detection", "tracking", "reid", "train"],
                    help="Metrics to summarize (default: detection tracking reid train)")
    ap.add_argument("--per-metric", type=str, default="",
                    help='Overrides like "detection=5,tracking=5,reid=2,train=0"')
    return ap.parse_args()

def normalize_key(k: str) -> str:
    k = k.lower()
    return SYNONYMS.get(k, k)

def parse_per_metric(s: str):
    """Parse 'a=1,b=2' into dict {'a':1,'b':2}."""
    out = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().lower()] = int(v.strip())
    return out

def load_and_drop(path: Path, metrics, drop_thresh):
    """
    Read the log and drop the first N occurrences per metric (lines are dropped
    if they contain any of the target metrics still under their drop thresholds).
    Returns {metric: [seconds,...]} AFTER dropping.
    """
    counts = defaultdict(int)
    kept_values = defaultdict(list)

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = KEYVAL.search(line)
            keys_in_line = []
            pairs_in_line = []  # (norm_key, float_val)

            if m:
                msg = m.group("msg")
                for k, v in PAIR.findall(msg):
                    nk = normalize_key(k)
                    try:
                        val = float(v)  # seconds
                    except ValueError:
                        continue
                    pairs_in_line.append((nk, val))
                    if nk in drop_thresh:
                        keys_in_line.append(nk)

            # Decide whether to drop this line
            to_drop = False
            for k in set(keys_in_line):
                if counts[k] < drop_thresh[k]:
                    to_drop = True

            # Update counts for all target metrics present in this line
            for k in set(keys_in_line):
                counts[k] += 1

            if to_drop:
                continue  # skip the entire line

            # Keep values from this line for requested metrics
            for nk, val in pairs_in_line:
                if nk in drop_thresh:  # only collect requested metrics
                    kept_values[nk].append(val)

    return kept_values, counts

def summarize_ms(samples_s):
    """Return (n, mean, min, max, std) in ms, or None if empty."""
    if not samples_s:
        return None
    xs = [v * 1000.0 for v in samples_s]  # seconds -> ms
    n = len(xs)
    mean = statistics.fmean(xs)
    vmin = min(xs)
    vmax = max(xs)
    std = statistics.stdev(xs) if n >= 2 else 0.0
    return n, mean, vmin, vmax, std

def main():
    args = parse_args()
    if not args.file.exists():
        raise SystemExit(f"Log file not found: {args.file}")

    metrics = [m.lower() for m in args.metrics]
    drop_thresh = {m: args.drop_first for m in metrics}
    drop_over = parse_per_metric(args.per_metric)
    drop_thresh.update({k: drop_over[k] for k in drop_over})  # apply overrides

    kept_values, counts_seen = load_and_drop(args.file, metrics, drop_thresh)

    # Print summary table
    print(f"\nFile: {args.file}")
    print("Dropping first occurrences (per metric):")
    for m in metrics:
        print(f"  - {m}: {drop_thresh[m]} (seen={counts_seen.get(m,0)})")

    header = f"\n{'metric':<12} {'n_used':>8} {'mean [ms]':>12} {'min [ms]':>12} {'max [ms]':>12} {'std [ms]':>12}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        stat = summarize_ms(kept_values.get(m, []))
        if stat is None:
            print(f"{m:<12} {'—':>8} {'—':>12} {'—':>12} {'—':>12} {'—':>12}")
        else:
            n, mean, vmin, vmax, std = stat
            print(f"{m:<12} {n:>8} {mean:>12.3f} {vmin:>12.3f} {vmax:>12.3f} {std:>12.3f}")
    print()

if __name__ == "__main__":
    main()


# python3 timing.py   --file /home/enrique/vision_ws/eval_results/eval_results.log   --drop-first 10   --metrics detection tracking reid train --per-metric "reid=0"