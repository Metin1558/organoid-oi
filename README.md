# organoid-oi v4 — Reflexive Word-Length Sentence Assembly

**A working, hardware-ready demonstration: an organoid's own firing intensity, calibrated by reward/penalty, correctly orders words by length — producing a real sentence.**

Metin (ORCID: 0009-0006-4635-405X) · v4 · August 2026

---

## What this is

The organoid's physical response scales with input drive — a longer word activates more electrodes, produces more spikes. That coupling is direct and needs no training. What *is* trained, via Three-Factor STDP and a local Hebbian readout, is the decision layer: it starts blind and learns, through reward and penalty, to read the organoid's own signal correctly. The result is a system that takes a scrambled set of words and assembles them into the correct sentence — and correctly inserts a new word into an existing one.

We call this **reflexive**, deliberately, following Kagan et al.'s Pong demonstration: the mechanism is not "the organoid understands sentences," any more than an autocomplete model "understands" the next word it predicts. The mechanism is simple and named plainly. The result is real.

**Verified in simulation:**
- Word length ↔ firing intensity: correlation **+0.999** (7/7 monotonic pairs)
- Readout calibration via reward/penalty: accuracy rises from near-chance to consistently correct sentence assembly
- Insertion of a new word into an existing sentence: correct in every tested case

---

## Status

| Component | Status |
|---|---|
| Core mechanism (length → drive → readout learning) | **Working, verified in simulation** |
| 32-electrode scaling (2 electrodes/letter, redundant) | **Done and calibrated** |
| Live panel (typewriter view + technical readout) | **Working**, auto-records video of each run |
| Backend-agnostic calibration tool (`calibrate.py`) | **Working** — same code runs on either backend |
| Hardware adapter (`hardware/finalspark_organoid.py`) | **Structurally complete, same interface as the simulation** — the three I/O methods are explicit placeholders pending FinalSpark API integration |
| Live tissue run | **Not yet performed** |

This is not a claim that the system has run on real tissue. It is a claim that the system is *ready to*, in the specific and limited sense that connecting it requires filling in a known, isolated API surface — not further research or redesign.

---

## Why this path, not categorical learning

Earlier versions of this project (documented in full below) attempted to demonstrate that Three-Factor STDP drives *categorical* learning — an organoid distinguishing arbitrary classes (fruit images, Morse-coded anagrams, Braille letters) with no shortcut available. Across three task designs and four difficulty levels down to the simplest possible categorical task, no configuration produced a replicated learning effect, after six independently found and corrected measurement defects — including a data-contamination bug that had briefly made a null result look like a validated positive one.

That negative result stands and is documented in full (v3.2 preprint, DOI below). It motivated this version's approach directly: rather than requiring the organoid to build new categorical structure from nothing, this system uses a relationship the tissue already has — response magnitude scaling with input drive — and trains only the readout to use it correctly. This is a smaller claim than categorical learning, made with the same rigor, and it works.

---

## Run it

```bash
pip install numpy scipy --break-system-packages

python calibrate.py                          # verify the dose-response relationship
python sentence_demo.py --sentence "The party ended sooner." --insert last
```

A browser panel opens automatically and records a video of the run, saved on completion.

### Against real hardware (once the API adapter is filled in)

```bash
python calibrate.py --backend hardware --electrodes-only --api-key ... --culture-id ...
python calibrate.py --backend hardware --api-key ... --culture-id ...
python sentence_demo.py --backend hardware --api-key ... --culture-id ... --sentence "..." --insert ...
```

---

## Architecture

```
sentence_demo.py                Main script — task, training loop, CLI, --backend switch
calibrate.py                     Backend-agnostic dose-response check
core/                            Three-Factor STDP, reward/penalty injection, closed loop
sim/oi_synth.py                  Synthetic organoid (simulation backend)
hardware/finalspark_organoid.py  Hardware adapter — same interface as sim/oi_synth.py
panel/                           Live two-section panel + automatic video recording
```

---

## History

Full, unabridged account of every prior version, correction, and negative result:

- **v1.0 → v2.0**: retracted attribution claim (freeze-weights control did not isolate the substrate's contribution)
- **v2.0 → v3.2**: found the original task solvable by an untrained network; replaced it; found no categorical-learning effect under rigorous, six-times-repeated verification
- **v3.2 → v4**: pivoted from categorical to reflexive demonstration; verified; hardware-adapter built

See `YOL_HARITASI.md` (or the v3.2/v4 preprints) for the complete record, including the six corrected defects. Nothing there is superseded by this version — it is the reason this version exists.

---

## Data, code, funding

Source at github.com/Metin1558/organoid-oi. Funded by an Emergent Ventures grant (Mercatus Center, George Mason University). Hardware coordination with FinalSpark's Neuroplatform is in progress; this version is the readiness package for that step.

---

*Not peer reviewed. v4 — August 2026.*
