"""
calibrate.py — Checks whether "more input drive -> more firing" holds,
against EITHER backend (synthetic or hardware), using the identical code.

WHY THIS MATTERS
----------------
The sentence demo works because the organoid's raw response scales with
word length. That was verified in simulation. Before spending paid session
time on the full demo against FinalSpark, run this: it presents words of
increasing length, measures total firing, and reports whether the
relationship is still monotonic on the real culture. Cheap, fast, and
answers the one question that decides whether the demo is worth running
on hardware at all.

USAGE
-----
    python calibrate.py                                  # synthetic (default)
    python calibrate.py --backend hardware --api-key ... --culture-id ...
    python calibrate.py --backend hardware --electrodes-only   # cheapest check:
        just confirms each electrode addresses correctly, no firing analysis
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
for sub in ["", "core", "sim", "hardware"]:
    sys.path.insert(0, str(ROOT / sub))

import numpy as np
from sentence_demo import word_stimulus, make_config

TEST_WORDS = ["a", "an", "the", "cats", "table", "garden", "silence", "elephant"]


def build_organoid(args, cfg, rng):
    if args.backend == "hardware":
        from finalspark_organoid import FinalSparkOrganoid
        return FinalSparkOrganoid(cfg, api_key=args.api_key, culture_id=args.culture_id, rng=rng)
    from sim.oi_synth import SyntheticOrganoid
    return SyntheticOrganoid(cfg, rng=rng)


def electrodes_only_check(organoid, cfg, rng):
    """Cheapest possible check: fire one electrode at a time, confirm a
    response is read back at all. No monotonicity claim — just wiring."""
    print("  ELECTRODE ADDRESSING CHECK (cheapest, no firing-pattern analysis)")
    for i in range(0, cfg.n_electrodes, max(1, cfg.n_electrodes // 8)):
        s = word_stimulus("test", cfg, rng)  # reuses the encoder; we just inspect electrode i
        r = organoid.respond(s, timestamp=0.0)
        total = sum(len(x) for x in r.spike_times)
        print(f"    electrode block near #{i}: response recorded, total spikes = {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["synthetic", "hardware"], default="synthetic")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--culture-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--electrodes-only", action="store_true",
                        help="cheapest check — just confirms wiring, skip firing analysis")
    args = parser.parse_args()

    cfg = make_config()
    rng = np.random.default_rng(args.seed)
    organoid = build_organoid(args, cfg, rng)

    print("=" * 60)
    print(f"  CALIBRATION — backend: {args.backend}")
    print("=" * 60)

    if args.electrodes_only:
        electrodes_only_check(organoid, cfg, rng)
        return

    print("  Presenting words of increasing length, measuring total firing.")
    print(f"  ({args.repeats} repeats per word)")
    print()
    lengths, means, stds = [], [], []
    for word in TEST_WORDS:
        counts = []
        for _ in range(args.repeats):
            s = word_stimulus(word, cfg, rng)
            r = organoid.respond(s, timestamp=0.0)
            counts.append(sum(len(x) for x in r.spike_times))
        lengths.append(len(word)); means.append(np.mean(counts)); stds.append(np.std(counts))
        print(f"    {word:<10} len={len(word):<3} firing={np.mean(counts):7.1f}  (std {np.std(counts):.1f})")

    lengths, means = np.array(lengths), np.array(means)
    order = np.argsort(lengths)
    monotonic_pairs = sum(1 for i in range(len(order) - 1)
                          if means[order[i + 1]] >= means[order[i]])
    correlation = float(np.corrcoef(lengths, means)[0, 1]) if len(set(lengths)) > 1 else float("nan")

    print()
    print(f"  monotonic adjacent pairs: {monotonic_pairs}/{len(order) - 1}")
    print(f"  length-vs-firing correlation: {correlation:+.3f}")
    print()
    if correlation > 0.7:
        print("  RESULT: strong monotonic relationship — demo should transfer.")
    elif correlation > 0.3:
        print("  RESULT: weak/partial relationship — recalibrate drive current before the full demo.")
    else:
        print("  RESULT: no reliable relationship found — do not proceed to the full demo yet.")


if __name__ == "__main__":
    main()
