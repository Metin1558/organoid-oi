"""
oi_run_monitored.py — Iki kollu oturumu calistirir, canli paneli besler.

Calistir:
    python oi_run_monitored.py
    python oi_run_monitored.py --trials 300 --seed 7

Tarayici kendiliginden acilir. Deney ilerledikce panel guncellenir.

NE YAPAR
--------
Iki kol calistirir:
    Kol A — plastisite acik   (learning_rate = config varsayilani)
    Kol B — kontrol           (learning_rate = 0)

Ayni tohum, ayni uyaran sirasi, ayni okuma katmani. Trial'lar donusumlu
gonderilir, boylece panelde iki egri es zamanli ilerler.

Ayrilabilirlik her 15 trial'da bir olculur — decoder'a bakmayan olcum budur.

CANLI DOKUYA GECERKEN
---------------------
Simulasyonda iki kol paralel calisir. Canli dokuda tek doku vardir, o yuzden
kollar ayni doku uzerinde donusumlu BLOK olarak calisir (ABAB). Panel ikisini
de ayni sekilde gosterir — degistirmen gereken tek sey bu dosyadaki dongu,
panel ve monitor aynen kalir.
"""

import argparse
import sys
from pathlib import Path

KOK = Path(__file__).parent.resolve()
for alt in ["", "core", "sim", "analysis", "monitor"]:
    sys.path.insert(0, str(KOK / alt))

import numpy as np

from core.oi_types import ExperimentConfig
from core.oi_signal import synthetic_stimulus
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid
from oi_monitor import Monitor

DESEN = ["center", "edge", "stripe"]


# ---------------------------------------------------------------
def hiz_vektoru(resp):
    return np.array([len(x) for x in resp.spike_times], dtype=float)


def ayrilabilirlik(org, cfg, tohum=999, tekrar=4):
    """Kategoriler-arasi mesafe / kategori-ici yayilim. Decoder'a bakmaz."""
    rng = np.random.default_rng(tohum)
    kat = {k: [] for k in cfg.categories}
    for i in range(tekrar * 3):
        k = cfg.categories[i % 3]
        r = org.respond(synthetic_stimulus(k, DESEN[i % 3], cfg, rng), timestamp=0.0)
        kat[k].append(hiz_vektoru(r))
    ort = {k: np.mean(v, axis=0) for k, v in kat.items()}
    ic = np.mean([np.mean([np.linalg.norm(v - ort[k]) for v in kat[k]])
                  for k in cfg.categories])
    ks = list(cfg.categories)
    dis = np.mean([np.linalg.norm(ort[ks[a]] - ort[ks[b]])
                   for a in range(3) for b in range(a + 1, 3)])
    return float(dis / (ic + 1e-9))


def elektrot_haritasi(resp, n=32):
    """Yaniti 32 elektrotluk 0/1 haritasina indirger (panel gosterimi icin)."""
    v = hiz_vektoru(resp)
    if len(v) == 0:
        return [0] * n
    # noron sayisi 32'den farkli olabilir — kovalara bol
    kova = np.array_split(v, n)
    return [1 if k.sum() > 0 else 0 for k in kova]


def piksel_haritasi(stim, n=64):
    """Uyarani 8x8 yogunluk haritasina indirger."""
    yog = np.array([len(s) for s in stim.spike_times], dtype=float)
    if yog.size == 0:
        return [0.0] * n
    kova = np.array_split(yog, n)
    d = np.array([k.mean() if k.size else 0.0 for k in kova])
    if d.max() > 0:
        d = d / d.max()
    return [float(x) for x in d]


# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0,
                    help="trial arasi bekleme (sn) — izlemeyi yavaslatmak icin")
    ap.add_argument("--sep-every", type=int, default=15)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    mon = Monitor(port=a.port, open_browser=not a.no_browser)
    mon.session(culture="synthetic", note="two-arm attribution session")

    # ---- iki kol, ayni tohum ----
    cfgA = ExperimentConfig()
    cfgB = ExperimentConfig(learning_rate=0.0)

    rngA = np.random.default_rng(a.seed)
    rngB = np.random.default_rng(a.seed)
    orgA = SyntheticOrganoid(cfgA, rng=rngA)
    orgB = SyntheticOrganoid(cfgB, rng=rngB)
    expA = ClosedLoopExperiment(orgA, cfgA, rng=rngA)
    expB = ClosedLoopExperiment(orgB, cfgB, rng=rngB)

    print("[run] %d trial, tohum %d — kol A plastisite acik, kol B kontrol"
          % (a.trials, a.seed))

    import time
    for i in range(a.trials):
        kat = cfgA.categories[i % 3]
        desen = DESEN[i % 3]

        sA = synthetic_stimulus(kat, desen, cfgA, rngA)
        sB = synthetic_stimulus(kat, desen, cfgB, rngB)

        mon.stimulus(kat, piksel_haritasi(sA))

        tA = expA.run_trial(sA, i)
        tB = expB.run_trial(sB, i)

        sepA = sepB = None
        if i % a.sep_every == 0:
            sepA = ayrilabilirlik(orgA, cfgA)
            sepB = ayrilabilirlik(orgB, cfgB)

        for kol, t, org, cfg, sep in (("A", tA, orgA, cfgA, sepA),
                                      ("B", tB, orgB, cfgB, sepB)):
            resp = getattr(t, "response", None)
            mon.trial(
                arm=kol, trial=i, category=kat,
                predicted=getattr(t, "predicted", None),
                correct=bool(getattr(t, "correct", False)),
                electrodes=elektrot_haritasi(resp) if resp is not None else None,
                spikes=int(sum(len(x) for x in resp.spike_times)) if resp is not None else 0,
                separability=sep,
            )

        if a.delay:
            time.sleep(a.delay)

    accA = 100 * np.mean([t.correct for t in expA.trial_history[-40:]])
    accB = 100 * np.mean([t.correct for t in expB.trial_history[-40:]])
    print("[run] bitti — kol A %.1f%%  kol B %.1f%%  fark %+.1f puan"
          % (accA, accB, accA - accB))

    mon.finish()


if __name__ == "__main__":
    main()
