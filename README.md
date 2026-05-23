# organoid-oi

**Closed-loop Organoid Intelligence system for visual categorical learning.**

A biologically grounded simulation framework implementing STDP-based learning in a synthetic brain organoid for visual category discrimination. Designed as a computational bridge toward real MEA hardware integration (CL1, HD-MEA).

Companion project to [organoid v6.3](https://github.com/metin/organoid-v63) — the electrophysiology analysis pipeline.

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
└── test_oi_v3.py  47 synthetic validation tests (100% pass)
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

### Rank-Order Coding (Van Rullen & Thorpe 2001)

Most salient pixel fires first (latency ≈ 0 ms), least salient fires last. Since STDP is exponentially sensitive to spike timing, the most prominent visual features drive learning most strongly — consistent with biological retinal processing.

### Frozen-weight validation

`freeze_weights_test()` freezes organoid weights and measures accuracy with the current decoder. If performance drops to chance (~33% for 3 categories), STDP is confirmed as the learning source — not the decoder.

---

## Installation

```bash
pip install numpy scipy Pillow
```

Python 3.10+ required. No GPU needed.

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

47 synthetic tests across 6 modules — 100% pass rate:

| Module | Tests | What is tested |
|--------|-------|----------------|
| oi_signal (DoG + rank-order) | 16 | DoG channels, latency ordering, pattern separation |
| oi_stdp (Three-Factor, vectorized) | 12 | LTP/LTD direction, neuromodulator scaling, speed |
| oi_reward (current injection) | 15 | Reward/penalty currents, LIF response, neuromod factors |
| oi_synth (LIF organoid) | — | Inherited from sim module |
| oi_loop (full pipeline) | 10 | Hebbian decoder, blanking, epoch homeostatic |
| Freeze-weights test | 4 | Validates STDP as learning source |

---

## Limitations

- **Synthetic organoid only.** Real hardware requires CL1 or HD-MEA with real-time readout API. `SyntheticOrganoid` implements the interface that real hardware would replace.
- **Ragged array outer loop.** Spike trains have variable length — the outer loop over neuron pairs is unavoidable without padding. For networks > 500 neurons, consider fixed-length spike matrices.
- **Hebbian decoder separability.** Local Hebbian learning may not separate linearly non-separable categories. For complex stimuli, a larger neuron count improves population coverage.
- **No long-term stability guarantee.** Epoch-based homeostatic (every 10 trials) prevents acute collapse but does not guarantee stability over thousands of trials.

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

*Companion to organoid v6.3. Not peer reviewed. May 2026.*
