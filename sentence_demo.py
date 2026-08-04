"""
sentence_demo.py — The organoid learns to sort words by LENGTH, reflexively.

PRINCIPLE
---------
This is not a categorical-learning claim. It mirrors Kagan et al.'s Pong
demonstration: a REFLEXIVE mechanism, not novel cognition.

  1. The organoid's PHYSICAL RESPONSE is naturally proportional to word
     length (no training needed) — a longer word drives more electrodes,
     produces more spikes. This is a direct physical coupling.
  2. The READOUT LAYER starts BLIND (random weights). Asked "is this word
     short or long", it starts near chance.
  3. Reward/penalty (Three-Factor STDP + local Hebbian decoder) teaches the
     readout to listen to the signal the organoid already carries.

Claim: not "the organoid wrote a sentence" — the organoid produces a
length-proportional physical response; the readout learns to read it;
the result is words correctly ordered by length. Comparable to an LLM
"predicting the next token" — the mechanism is simple, the outcome is real.

USAGE
-----
    python sentence_demo.py --sentence "The party ended sooner." --insert last
    python sentence_demo.py --sentence "The storm brings sadness." --insert real
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
for sub in ["", "core", "sim"]:
    sys.path.insert(0, str(ROOT / sub))

import numpy as np
from core.oi_types import ExperimentConfig, StimulusPattern
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid

sys.path.insert(0, str(ROOT / "panel"))


def word_stimulus(word, cfg, rng):
    """
    Word length -> drive strength. Each LETTER fires TWO electrodes
    (redundant pair), so a 32-electrode array supports words up to 16
    letters — comfortably more than any word used here — and the
    redundancy gives a stronger, more robust signal per letter than a
    single electrode would (useful headroom for noisier real tissue).
    """
    n_el = cfg.n_electrodes
    duration = cfg.stimulus_duration_s
    per_letter = max(1, n_el // 16)  # electrodes committed per letter position
    length = min(len(word), n_el // per_letter)
    traces = [np.array([]) for _ in range(n_el)]
    for letter_idx in range(length):
        t = duration * (0.2 + 0.5 * letter_idx / max(length - 1, 1)) + rng.normal(0, 0.002)
        t = max(0.0, min(t, duration - 1e-4))
        for k in range(per_letter):
            e = letter_idx * per_letter + k
            if e < n_el:
                traces[e] = np.array([t + rng.normal(0, 0.0005)])
    return StimulusPattern(spike_times=traces, n_electrodes=n_el, duration_s=duration, label=word)


def make_config(**kw):
    kw.setdefault("categories", ["short", "long"])
    kw.setdefault("n_electrodes", 32)        # matches FinalSpark's commonly cited channel count
    kw.setdefault("r_membrane_mohm", 30.0)   # recalibrated for 32 electrodes, 2-per-letter redundancy
    kw.setdefault("v_thresh_jitter_mv", 5.0)
    kw.setdefault("decoder_lr", 0.05)
    return ExperimentConfig(**kw)


def train(organoid, cfg, word_pool, threshold, rng, n_trials=200, window=20, on_event=None):
    """
    Trains the readout via reward/penalty. Each trial: a random word is
    drawn, labelled 'short'/'long' against the threshold, and the closed
    loop (STDP + local Hebbian decoder update) processes it. Returns a
    windowed accuracy curve — starts near chance, should trend upward.
    """
    exp = ClosedLoopExperiment(organoid, cfg, rng=rng)
    history, curve = [], []
    for i in range(n_trials):
        word = rng.choice(word_pool)
        stim = word_stimulus(word, cfg, rng)
        stim.label = "long" if len(word) >= threshold else "short"
        trial = exp.run_trial(stim, i)
        history.append(int(trial.correct))
        if on_event:
            on_event({"type": "trial", "step": i, "word": word,
                      "label": stim.label, "correct": bool(trial.correct)})
        if (i + 1) % window == 0:
            recent = history[-window:]
            curve.append((i + 1, sum(recent) / len(recent)))
    return exp, curve


def readout_score(exp, organoid, cfg, word, rng, repeats=3):
    """
    The TRAINED decoder's continuous 'long' activation, minus 'short'.
    Before training this is near-random; after training it should
    correlate with word length.
    """
    scores = []
    for _ in range(repeats):
        stim = word_stimulus(word, cfg, rng)
        response = organoid.respond(stim, timestamp=0.0)
        rates = exp.decoder.firing_rates(response)
        activation = exp.decoder.weights @ rates
        i_long = exp.decoder.categories.index("long")
        i_short = exp.decoder.categories.index("short")
        scores.append(float(activation[i_long] - activation[i_short]))
    return float(np.mean(scores))


def write_sentence(exp, organoid, cfg, words, rng):
    scores = {w: readout_score(exp, organoid, cfg, w, rng) for w in words}
    order = sorted(range(len(words)), key=lambda i: (scores[words[i]], i))
    return [words[i] for i in order], scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default="The party ended sooner.")
    parser.add_argument("--insert", type=str, default="last")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-panel", action="store_true")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--backend", choices=["synthetic", "hardware"], default="synthetic",
                        help="synthetic = simulation (default). hardware = FinalSpark "
                             "Neuroplatform — requires hardware/finalspark_organoid.py "
                             "to be filled in with real API calls first (see its docstring).")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--culture-id", type=str, default=None)
    args = parser.parse_args()

    panel = None
    if not args.no_panel:
        import server as panel
        panel.start(port=args.port)
        import time as _time
        _time.sleep(1.0)

    words = [w.strip(".,!?") for w in args.sentence.split()]
    lengths = [len(w) for w in words]
    threshold = int(np.median(lengths))

    print("=" * 64)
    print("  SENTENCE DEMO — readout learns word length via reward/penalty")
    print("=" * 64)
    print(f"  words: {words}")
    print(f"  lengths: {lengths}  (short/long threshold: {threshold} letters)")
    print()

    cfg = make_config(learning_rate=0.02)
    rng = np.random.default_rng(args.seed)
    if args.backend == "hardware":
        sys.path.insert(0, str(ROOT / "hardware"))
        from finalspark_organoid import FinalSparkOrganoid
        organoid = FinalSparkOrganoid(cfg, api_key=args.api_key, culture_id=args.culture_id, rng=rng)
        print("  [backend] FinalSpark hardware — see hardware/finalspark_organoid.py")
    else:
        organoid = SyntheticOrganoid(cfg, rng=rng)

    if panel:
        panel.push(phase="training", words=words, lengths=dict(zip(words, lengths)))

    def on_event(ev):
        if panel:
            panel.push(phase="training", last_event=ev)

    print("  TRAINING (readout starts blind, calibrates via reward/penalty)")
    pool = words + [args.insert]
    exp, curve = train(organoid, cfg, pool, threshold, rng, n_trials=args.trials, on_event=on_event)
    print("  step      accuracy")
    for step, acc in curve:
        bar = "#" * int(acc * 30)
        print(f"   {step:4d}     {acc:5.1%}  {bar}")
        if panel:
            panel.push(curve=[list(c) for c in curve[:curve.index((step, acc)) + 1]])
    print()

    after, after_scores = write_sentence(exp, organoid, cfg, words, np.random.default_rng(1000))
    print("  AFTER training — sentence written by the organoid:")
    print("   ", " ".join(after) + ".")
    print("  (readout scores: %s)" % {k: round(v, 2) for k, v in after_scores.items()})
    if panel:
        panel.push(phase="complete", current_sentence=after)
        import time as _time
        _time.sleep(0.5)

    print()
    print(f"  INSERTING NEW WORD: '{args.insert}' (length {len(args.insert)})")
    all_words = words + [args.insert]
    final, final_scores = write_sentence(exp, organoid, cfg, all_words, np.random.default_rng(1001))
    print("  UPDATED SENTENCE:")
    print("   ", " ".join(final) + ".")
    print("  (readout scores: %s)" % {k: round(v, 2) for k, v in final_scores.items()})

    if panel:
        panel.push(current_sentence=final, finished=True)
        print()
        print("  [panel] recording will auto-save now — leave the browser tab open a moment.")
        import time as _time
        _time.sleep(3.0)


if __name__ == "__main__":
    main()
