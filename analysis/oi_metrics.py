"""
oi_metrics.py
=============
Learning metrics for closed-loop OI experiments.

Computes:
    - Overall and per-category accuracy
    - Learning curve (rolling window accuracy)
    - STTC delta: synchrony change across training
    - Weight matrix evolution statistics
    - Trial-by-trial performance summary

Imports STTC from organoid v6.3 if available.
Falls back to internal implementation if not found.

No interpretation — returns numbers and arrays only.
"""

import numpy as np
from typing import List, Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import STTC from organoid v6.3
_V63_PATH = Path(__file__).parent.parent.parent / 'organoid_v6_3'
_HAS_V63 = False
if _V63_PATH.exists():
    sys.path.insert(0, str(_V63_PATH))
    try:
        from organoid_metrics import sttc_iki_kanal as sttc_pair
        _HAS_V63 = True
    except ImportError:
        pass

if not _HAS_V63:
    # Fallback: internal STTC implementation (Cutts & Eglen 2014)
    def sttc_pair(train_a, train_b, duration_s, dt=0.01):
        """Minimal STTC implementation — fallback if v6.3 not available."""
        def tile_time(spikes, dur, dt):
            if len(spikes) == 0:
                return 0.0
            total = 0.0
            for t in spikes:
                total += min(dt, t) + min(dt, dur - t)
            return min(total / dur, 1.0)

        def prop_in_window(spikes_a, spikes_b, dt):
            if len(spikes_a) == 0:
                return 0.0
            count = 0
            for t in spikes_a:
                if np.any(np.abs(spikes_b - t) <= dt):
                    count += 1
            return count / len(spikes_a)

        if len(train_a) == 0 or len(train_b) == 0:
            return 0.0

        T_A = tile_time(train_a, duration_s, dt)
        T_B = tile_time(train_b, duration_s, dt)
        P_A = prop_in_window(train_a, train_b, dt)
        P_B = prop_in_window(train_b, train_a, dt)

        denom_1 = 1 - P_A * T_B
        denom_2 = 1 - P_B * T_A
        if denom_1 == 0 or denom_2 == 0:
            return 1.0

        return 0.5 * ((P_A - T_B) / denom_1 + (P_B - T_A) / denom_2)


# ─────────────────────────────────────────────────────────────
# ACCURACY METRICS
# ─────────────────────────────────────────────────────────────

def accuracy(trials: list) -> float:
    """Overall fraction of correct trials."""
    if not trials:
        return 0.0
    return sum(t.correct for t in trials) / len(trials)


def per_category_accuracy(trials: list, categories: List[str]) -> Dict[str, float]:
    """Accuracy per category."""
    result = {}
    for cat in categories:
        cat_trials = [t for t in trials if t.stimulus.label == cat]
        if cat_trials:
            result[cat] = sum(t.correct for t in cat_trials) / len(cat_trials)
        else:
            result[cat] = float('nan')
    return result


def learning_curve(trials: list, window: int = 20) -> np.ndarray:
    """
    Rolling accuracy over trial sequence.
    Window size controls smoothing — larger = smoother but slower to detect changes.
    """
    if not trials:
        return np.array([])
    correct = np.array([t.correct for t in trials], dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(correct, kernel, mode='same')


def chance_level(n_categories: int) -> float:
    """Theoretical chance accuracy for uniform random responding."""
    return 1.0 / n_categories


def above_chance(trials: list, n_categories: int, margin: float = 0.05) -> bool:
    """
    Returns True if accuracy exceeds chance by margin.
    No statistical test — just threshold comparison.
    """
    return accuracy(trials) > chance_level(n_categories) + margin


# ─────────────────────────────────────────────────────────────
# STTC METRICS
# ─────────────────────────────────────────────────────────────

def mean_sttc(spike_times: List[np.ndarray], duration_s: float, dt: float = 0.01) -> float:
    """
    Mean STTC across all neuron pairs in a response.

    Args:
        spike_times : list of spike time arrays, one per neuron
        duration_s  : recording duration
        dt          : STTC coincidence window (seconds)

    Returns:
        mean STTC across all pairs (float)
    """
    n = len(spike_times)
    if n < 2:
        return float('nan')

    values = []
    for i in range(n):
        for j in range(i + 1, n):
            val = sttc_pair(spike_times[i], spike_times[j], duration_s, dt)
            values.append(val)

    return float(np.mean(values)) if values else float('nan')


def sttc_trajectory(trials: list, step: int = 10) -> Dict[str, np.ndarray]:
    """
    STTC over training — sampled every `step` trials.

    Returns:
        dict with 'trial_ids' and 'sttc_values' arrays.
    """
    trial_ids = []
    sttc_values = []

    for i in range(0, len(trials), step):
        trial = trials[i]
        sttc_val = mean_sttc(
            trial.response.spike_times,
            trial.response.duration_s,
        )
        trial_ids.append(i)
        sttc_values.append(sttc_val)

    return {
        'trial_ids': np.array(trial_ids),
        'sttc_values': np.array(sttc_values),
    }


# ─────────────────────────────────────────────────────────────
# WEIGHT METRICS
# ─────────────────────────────────────────────────────────────

def weight_divergence(
    weights_before: np.ndarray,
    weights_after: np.ndarray,
) -> dict:
    """
    Measure how much weights changed during training.

    Returns:
        dict with mean_change, max_change, fraction_changed (> 0.01)
    """
    diff = weights_after - weights_before
    return {
        'mean_abs_change': float(np.abs(diff).mean()),
        'max_abs_change': float(np.abs(diff).max()),
        'fraction_changed': float((np.abs(diff) > 0.01).mean()),
        'net_change': float(diff.mean()),  # positive = net potentiation
    }


# ─────────────────────────────────────────────────────────────
# EXPERIMENT SUMMARY
# ─────────────────────────────────────────────────────────────

def experiment_summary(trials: list, config) -> dict:
    """
    Full descriptive summary of experiment results.
    No interpretation — all numbers.
    """
    if not trials:
        return {'n_trials': 0}

    n = len(trials)
    cats = config.categories
    chance = chance_level(len(cats))

    return {
        'n_trials': n,
        'n_categories': len(cats),
        'chance_level': chance,
        'overall_accuracy': accuracy(trials),
        'per_category_accuracy': per_category_accuracy(trials, cats),
        'first_quarter_accuracy': accuracy(trials[:n//4]),
        'last_quarter_accuracy': accuracy(trials[3*n//4:]),
        'n_correct': sum(t.correct for t in trials),
        'n_incorrect': sum(not t.correct for t in trials),
        'reward_counts': {
            sig: sum(t.reward.signal_type == sig for t in trials)
            for sig in ['reward', 'penalty', 'neutral']
        },
        'mean_confidence': float(np.mean([
            t.response.confidence for t in trials
            if t.response.confidence is not None
        ])),
    }
