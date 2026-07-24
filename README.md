# organoid-oi

**Closed-loop Organoid Intelligence system for visual categorical learning.**

A biologically grounded simulation framework implementing STDP-based learning in a synthetic brain organoid for visual category discrimination. Designed as a computational bridge toward real MEA hardware integration.

Companion project to [Axon](https://github.com/Metin1558/axon) — the organoid MEA electrophysiology analysis pipeline.

**Current version:** v3.1 (July 2026) · **Tests:** 47/47 passing

> **v3.0 correction.** v3.0 was published claiming 47/47. Two of those tests were
> later found to be measuring the wrong quantity and, on re-audit, v3.0 should be
> read as **45/47**. Both tests were rewritten in v3.1. No framework code was
> changed — the defects were in the tests, not the mechanism.
> See [CHANGELOG_v3.1.md](CHANGELOG_v3.1.md) for the full account.

---

## Overview

The system presents visual stimuli to a simulated organoid via electrode activation patterns, reads the neural response, delivers biologically realistic reward/penalty signals as current injection, and updates synaptic weights via Three-Factor STDP. A local Hebbian readout layer decodes the organoid's population activity.

```
Image → DoG filter → Rank-order spikes → Organoid (LIF neurons)
                                               ↓
                                         Decode response
                                               ↓
                              Reward (10 Hz theta) or Penalty (200 Hz noise)
                                               ↓
                                    Current injection → Phase 2
                                               ↓
                              Three-Factor STDP + Homeostatic scaling
```

---

## Architecture

```
core/
├── oi_types.py    Shared dataclasses (StimulusPattern, OrganoidResponse, ...)
├── oi_signal.py   Image → spike pattern (DoG filter, rank-order latency coding)
├── oi_stdp.py     Three-Factor STDP (Bi & Poo 1998, vectorized)
├── oi_reward.py   Reward/penalty as current injection (not external weight forcing)
└── oi_loop.py     Closed-loop orchestrator + Hebbian decoder + blanking period

sim/
└── oi_synth.py    Synthetic organoid (Leaky Integrate-and-Fire neurons)

analysis/
└── oi_metrics.py  Learning metrics (accuracy, STTC, weight divergence)

tests/
└── test_oi_v3.py  47 validation tests across 6 suites
```

---

## Key Design Decisions

### No mathematical shortcuts

Every mechanism in this system has a biological counterpart:

| Component | v1 (shortcut) | v3 (biological) |
|-----------|--------------|-----------------|
| Reward | `dw * reward_scale` — forced direction | 10 Hz theta current → natural LTP |
| Penalty | `dw * -0.5` — forced direction | 200 Hz noise current → natural LTD |
| Decoder | Trained delta rule | Local Hebbian with weight decay |
| Homeostatic | Every trial | Every 10 trials (slow timescale) |

### Three-Factor STDP (Frémaux & Gerstner 2016)

`dw = neuromodulator × STDP(dt)`

Neuromodulator scales magnitude — spike timing still drives direction. Reward/penalty delivered as actual current to LIF neurons (Phase 2), which shapes post-synaptic timing naturally via STDP, without any external override.

Verified by paired control: under the default configuration, disabling STDP
(`learning_rate=0.0`) reduces the weight change to exactly `0.0` on all seeds
tested, while the enabled arm produces non-zero change. The contribution is
attributable to STDP and not to homeostatic rescaling.

### Rank-Order Coding (Van Rullen & Thorpe 2001)

Most salient pixel fires first (latency ≈ 0 ms), least salient fires last. Since STDP is exponentially sensitive to spike timing, the most prominent visual features drive learning most strongly — consistent with biological retinal processing.

### Frozen-weight validation

`freeze_weights_test()` freezes organoid weights and measures accuracy with the current decoder. Performance dropping to chance (~33% for 3 categories) is evidence that learning is not carried by the decoder alone.

Note that this control **does not transfer to live tissue** — biological synapses cannot be frozen. A closed-loop hardware experiment needs a different control for the same claim (for example, a reward-decoupled or no-injection arm run interleaved within the same session).

---

## Installation

```bash
pip install numpy scipy Pillow
```

Python 3.10+ required. No GPU needed.

`scipy` is used for DoG convolution and `Pillow` for image loading; both are
imported lazily, so the test suite itself runs on `numpy` alone.

---

## Quick Start

```bash
# Run validation tests
python oi_cli.py test

# 50-trial demo
python oi_cli.py demo

# Full 300-trial experiment
python oi_cli.py run --trials 300

# Custom
python oi_cli.py run --trials 500 --neurons 150 --lr 0.005 --quiet
```

---

## Validation

47 tests across 6 suites — all passing as of v3.1. Runtime ~2 s, deterministic.

| Suite | Tests | What is tested |
|-------|-------|----------------|
| T1 — Vectorized STDP | 12 | LTP/LTD direction, neuromodulator scaling, vectorization correctness and speed |
| T2 — Epoch-based homeostatic | 5 | Epoch boundary triggering, trial counting, STDP survival under rescaling |
| T3 — Blanking period | 7 | Stimulus-artifact suppression, spike retention outside the blank window |
| T4 — Hebbian decoder | 9 | Initialization, decoding, local update direction, decay-bounded weights |
| T5 — Full closed-loop | 10 | End-to-end trial, reward delivery, history, accuracy bookkeeping |
| T6 — Freeze-weights | 4 | Decoder-only performance falls to chance; state preserved |

Two assertions were rewritten in v3.1:

- **T2 washout** — v3.0 used an absolute threshold on an 8-electrode / 5-neuron configuration in which the LIF population emits zero spikes, so STDP contributed exactly nothing and the test was measuring homeostatic drift. Replaced with a paired STDP-on / STDP-off control.
- **T4 decay bound** — v3.0 asserted `max|w| < 50` under a drive whose analytical equilibrium is `lr·rate/decay = 10000`, a bound unreachable by construction. Replaced with a convergence check against the predicted equilibrium.

Passing a test and the underlying claim being true are separate things. Both rewrites were calibrated against measured values rather than chosen to make the suite green; the measurements and thresholds are recorded in the changelog and inline in the test file.

---

## Limitations

- **Synthetic organoid only.** Real hardware requires a platform with a real-time readout API. `SyntheticOrganoid` implements the three-method interface (`stimulate`, `read_spikes`, `inject_current`) that real hardware would replace; no other module needs modification.

- **Electrode count mismatch with target hardware.** Under default LIF parameters the population is *completely silent* below 64 electrodes — 0 spikes in 8/8 trials at 8, 16 and 32 electrodes. Target MEA platforms commonly provide 32. Either the LIF parameters or the input encoding must be revisited before hardware integration. Unresolved.

- **Empty trials.** Under the default 64-electrode configuration, 4–7 trials in 20 produce zero organoid spikes. A trial with no spikes gives the decoder nothing to read. A minimum-spike criterion for trial validity should be fixed *before* any live-tissue experiment, otherwise reported accuracy tracks firing rate rather than learning.

- **Firing-rate regime.** The simulation runs at ~0.7–0.9 Hz per neuron, broadly comparable to published human organoid medians (~0.42 Hz). It has not been validated at the low end of the observed range, where a 0.5 s readout window yields on the order of one spike per trial.

- **Ragged array outer loop.** Spike trains have variable length — the outer loop over neuron pairs is unavoidable without padding. For networks > 500 neurons, consider fixed-length spike matrices.

- **Hebbian decoder separability.** Local Hebbian learning may not separate linearly non-separable categories. For complex stimuli, a larger neuron count improves population coverage.

- **No long-term stability guarantee.** Epoch-based homeostatic scaling (every 10 trials) prevents acute collapse but does not guarantee stability over thousands of trials.

- **Not peer reviewed.**

---

## References

- Bi & Poo (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464–10472.
- Frémaux & Gerstner (2016). Neuromodulated STDP. *Frontiers in Neural Circuits*, 9, 85.
- Kagan et al. (2022). In vitro neurons learn and exhibit sentience when embodied in a simulated game-world. *Neuron*, 110(23), 3952–3969.
- Masquelier & Thorpe (2007). Unsupervised learning of visual features through STDP. *PLoS Computational Biology*, 3(2), e31.
- Song, Miller & Abbott (2000). Competitive Hebbian learning through STDP. *Nature Neuroscience*, 3(9), 919–926.
- Turrigiano (2008). The self-tuning neuron. *Cell*, 135(3), 422–435.
- Van Rullen & Thorpe (2001). Rate coding vs temporal order coding. *Neural Computation*, 13(6), 1255–1283.
- Abbott (1999). Lapicque's introduction of the integrate-and-fire model. *Brain Research Bulletin*, 50(5–6), 303–304.

---

*Companion to Axon v6.3. Not peer reviewed. v3.1 — July 2026.*
