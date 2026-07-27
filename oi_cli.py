"""
oi_cli.py — organoid-oi v3
===========================
Command Line Interface

Usage:
    python oi_cli.py test                         # Run validation tests
    python oi_cli.py demo                         # 50-trial demo
    python oi_cli.py run                          # 300-trial experiment
    python oi_cli.py run --trials 500 --neurons 150 --quiet

Commands:
    test    Run 47 synthetic validation tests
    demo    Quick 50-trial demo with verbose output
    run     Full experiment with configurable parameters
"""

import sys
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from core.oi_types import ExperimentConfig
from core.oi_signal import synthetic_stimulus
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid
from analysis.oi_metrics import experiment_summary, chance_level


def cmd_test(args):
    """Run synthetic validation tests."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tests' / 'test_oi_v3.py')],
        cwd=str(ROOT)
    )
    sys.exit(result.returncode)


def cmd_demo(args):
    """Quick 50-trial demo with verbose output."""
    print("=" * 65)
    print("  organoid-oi v3 — Demo (50 trials, synthetic organoid)")
    print("=" * 65)

    config = ExperimentConfig(
        image_size=4, n_electrodes=16, n_neurons=30,
        stimulus_duration_s=0.3, reward_duration_s=0.15,
        categories=['banana', 'apple', 'pear'],
        homeostatic=True, homeostatic_epoch=10,
        blanking_enabled=True, blanking_ms=5.0,
        use_rank_order=True, random_seed=42,
    )

    rng = np.random.default_rng(42)
    organoid = SyntheticOrganoid(config, rng=rng)
    exp = ClosedLoopExperiment(organoid, config, rng=rng)

    patterns = ['center', 'edge', 'stripe']
    labels = config.categories

    print(f"\n  Categories : {config.categories}")
    print(f"  Chance     : {chance_level(len(config.categories)):.1%}")
    print(f"  Neurons    : {config.n_neurons}")
    print(f"  Electrodes : {config.n_electrodes}")
    print(f"  Encoding   : rank-order (DoG)")
    print(f"  Homeostatic: every {config.homeostatic_epoch} trials")
    print(f"  Blanking   : {config.blanking_ms} ms")
    print()

    for i in range(50):
        lbl = labels[i % 3]
        pat = patterns[i % 3]
        stim = synthetic_stimulus(lbl, pat, config, rng)
        exp.run_trial(stim, trial_id=i, verbose=True)

    summary = experiment_summary(exp.trial_history, config)
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  Overall accuracy   : {summary['overall_accuracy']:.1%}")
    print(f"  Chance level       : {chance_level(3):.1%}")
    print(f"  First 25 trials    : {summary['first_quarter_accuracy']:.1%}")
    print(f"  Last 25 trials     : {summary['last_quarter_accuracy']:.1%}")
    print()
    print("  Per-category:")
    for cat, acc in summary['per_category_accuracy'].items():
        print(f"    {cat:10s}: {acc:.1%}")
    print()
    print("  Reward counts:")
    for sig, n in summary['reward_counts'].items():
        print(f"    {sig:10s}: {n}")
    print("=" * 65)


def cmd_run(args):
    """Full configurable experiment."""
    print("=" * 65)
    print(f"  organoid-oi v3 — Experiment")
    print(f"  Trials      : {args.trials}")
    print(f"  Categories  : {args.categories}")
    print(f"  Neurons     : {args.neurons}")
    print(f"  Homeostatic : every {args.homeostatic_epoch} trials")
    print(f"  Blanking    : {args.blanking_ms} ms")
    print("=" * 65)

    config = ExperimentConfig(
        image_size=args.image_size,
        n_electrodes=args.image_size ** 2,
        n_neurons=args.neurons,
        stimulus_duration_s=args.duration,
        reward_duration_s=args.reward_duration,
        categories=args.categories,
        homeostatic=True,
        homeostatic_epoch=args.homeostatic_epoch,
        blanking_enabled=not args.no_blanking,
        blanking_ms=args.blanking_ms,
        use_rank_order=not args.rate_coding,
        learning_rate=args.lr,
        random_seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    organoid = SyntheticOrganoid(config, rng=rng)
    exp = ClosedLoopExperiment(organoid, config, rng=rng)

    patterns = ['center', 'edge', 'stripe', 'random']
    labels = config.categories
    verbose = not args.quiet

    for i in range(args.trials):
        lbl = labels[i % len(labels)]
        pat = patterns[i % len(patterns)]
        stim = synthetic_stimulus(lbl, pat, config, rng)
        exp.run_trial(stim, trial_id=i, verbose=verbose)

        if (i + 1) % 50 == 0 and not verbose:
            acc = exp.accuracy(last_n=50)
            w_stats = organoid.weight_stats()
            print(f"  Trial {i+1:4d} | "
                  f"last-50 acc: {acc:.1%} | "
                  f"w_mean: {w_stats['mean']:.3f} | "
                  f"w_col_sum: {w_stats['mean_col_sum']:.2f}")

    summary = experiment_summary(exp.trial_history, config)
    print()
    print("=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"  Trials             : {summary['n_trials']}")
    print(f"  Overall accuracy   : {summary['overall_accuracy']:.1%}")
    print(f"  Chance level       : {chance_level(len(config.categories)):.1%}")
    print(f"  First quarter      : {summary['first_quarter_accuracy']:.1%}")
    print(f"  Last quarter       : {summary['last_quarter_accuracy']:.1%}")
    print()
    print("  Per-category accuracy:")
    for cat, acc in summary['per_category_accuracy'].items():
        print(f"    {cat:12s}: {acc:.1%}")
    print()
    print("  Decoder stats:")
    dec_stats = exp.decoder.weight_stats()
    print(f"    Updates     : {dec_stats['updates']}")
    print(f"    Max |weight|: {dec_stats['max_abs']:.4f}")
    print()
    print("  Organoid weight stats:")
    w = organoid.weight_stats()
    print(f"    Mean        : {w['mean']:.4f}")
    print(f"    Col sum mean: {w['mean_col_sum']:.4f}")
    print(f"    Saturated ↑ : {w['fraction_saturated_high']:.1%}")
    print(f"    Saturated ↓ : {w['fraction_saturated_low']:.1%}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description='organoid-oi v3 — Closed-Loop Organoid Intelligence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True

    # test
    subparsers.add_parser('test', help='Run 47 synthetic validation tests')

    # demo
    subparsers.add_parser('demo', help='50-trial verbose demo')

    # run
    p = subparsers.add_parser('run', help='Full experiment')
    p.add_argument('--trials', type=int, default=300)
    p.add_argument('--categories', nargs='+', default=['banana', 'apple', 'pear'])
    p.add_argument('--neurons', type=int, default=100)
    p.add_argument('--image-size', type=int, default=8, dest='image_size')
    p.add_argument('--duration', type=float, default=0.5)
    p.add_argument('--reward-duration', type=float, default=0.3, dest='reward_duration')
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--homeostatic-epoch', type=int, default=10, dest='homeostatic_epoch')
    p.add_argument('--blanking-ms', type=float, default=5.0, dest='blanking_ms')
    p.add_argument('--no-blanking', action='store_true', dest='no_blanking')
    p.add_argument('--rate-coding', action='store_true', dest='rate_coding',
                   help='Use Poisson rate coding instead of rank-order')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    if args.command == 'test':
        cmd_test(args)
    elif args.command == 'demo':
        cmd_demo(args)
    elif args.command == 'run':
        cmd_run(args)


if __name__ == '__main__':
    main()

