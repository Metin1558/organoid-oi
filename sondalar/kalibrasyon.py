"""
kalibrasyon.py — Panel ogrenmeyi gorebiliyor mu?

SORUN
-----
Panel bir olcu aleti. Canli dokuya baglamadan once sunu bilmek zorundayiz:
bu alet, olcmesi gereken seyi gorebiliyor mu?

Su an goremiyor olabilir. Oyleyse organoidi baglayip iki cizgiyi ust uste
gordugumuzde, "organoid ogrenmedi" mi yoksa "panel kor" mu ayiramayiz.

YONTEM
------
Panele, ogrenip ogrenmedigini ONCEDEN BILDIGIMIZ iki durum veriyoruz:

  --pozitif   Kesin ogrenen bir organoid. Ogrenme STDP'den gelmiyor,
              dogrudan koda yazilmis. Yer gercegi bizde.
              Panel BU DURUMDA cizgileri AYIRMALI.

  --negatif   Kesin ogrenmeyen bir organoid. Yanitlari hic degismiyor.
              Panel BU DURUMDA cizgileri AYIRMAMALI.

Ikisi de gecerse panel calisiyor demektir ve canli dokuya guvenle baglanir.
Pozitif gecmezse panel kor. Negatif gecmezse panel yanlis alarm veriyor.

NOT
---
Buradaki "ogrenme" senaryodur, bulgu degildir. Amac organoidi degil
PANELI sinamaktir. Bu dosya hicbir bilimsel iddia uretmez.

KULLANIM
--------
    python sondalar/kalibrasyon.py --pozitif
    python sondalar/kalibrasyon.py --negatif
    python sondalar/kalibrasyon.py --pozitif --no-panel     (sadece terminal)
"""

import argparse
import sys
import time
from pathlib import Path

KOK = Path(__file__).parent.parent.resolve()
for alt in ["", "core", "sim", "analysis", "monitor"]:
    sys.path.insert(0, str(KOK / alt))

import numpy as np

from core.oi_types import ExperimentConfig, OrganoidResponse
from core.oi_signal import synthetic_stimulus
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid

DESEN = ["center", "edge", "stripe"]


# ===================================================================
class KalibrasyonOrganoid(SyntheticOrganoid):
    """
    Yer gercegi bilinen organoid.

    ogrenir=True  -> her kategoriye verdigi yanit, trial gectikce o
                     kategoriye ozgu hale gelir. Ogrendigini BILIYORUZ.
    ogrenir=False -> yanit hic degismez. Ogrenmedigini BILIYORUZ.

    Bu bir model degil, bir test sinyalidir. Osiloskobu sinamak icin
    bilinen bir dalga uretmek gibi.
    """

    def __init__(self, config=None, rng=None, ogrenir=False, olgunlasma=60.0):
        super().__init__(config, rng=rng)
        self.ogrenir = ogrenir
        self.olgunlasma = olgunlasma
        self._sayac = 0
        n = self.n_neurons
        # her kategoriye ayrilmis, ortusmeyen noron kumeleri
        self._kume = {}
        for i, k in enumerate(self.config.categories):
            self._kume[k] = np.arange(n)[(np.arange(n) + i) % 3 == 0]

    def respond(self, stimulus, timestamp, post_stimulus_current=None, dt_ms=1.0):
        self._sayac += 1
        n = self.n_neurons
        sure = self.config.stimulus_duration_s
        etiket = getattr(stimulus, "label", None) or self.config.categories[0]

        # olgunlasma: 0 -> hepsi ayni, 1 -> tam kategoriye ozgu
        if self.ogrenir:
            m = 1.0 - np.exp(-self._sayac / self.olgunlasma)
        else:
            m = 0.0

        kendi = set(self._kume.get(etiket, []).tolist())
        atesler = []
        for j in range(n):
            if j in kendi:
                p = 0.45 + 0.50 * m          # kendi kumesi guclenir
            else:
                p = 0.45 - 0.33 * m          # digerleri zayiflar
            sayi = self.rng.poisson(max(0.02, p) * 3.0)
            if sayi > 0:
                t = np.sort(self.rng.random(sayi) * sure)
                atesler.append(t)
            else:
                atesler.append(np.array([]))

        return OrganoidResponse(
            spike_times=atesler,
            n_neurons=n,
            duration_s=sure,
            timestamp=timestamp,
        )


# ===================================================================
def ayrilabilirlik(org, cfg, tohum=999, tekrar=4):
    rng = np.random.default_rng(tohum)
    kat = {k: [] for k in cfg.categories}
    for i in range(tekrar * 3):
        k = cfg.categories[i % 3]
        r = org.respond(synthetic_stimulus(k, DESEN[i % 3], cfg, rng), timestamp=0.0)
        kat[k].append(np.array([len(x) for x in r.spike_times], dtype=float))
    ort = {k: np.mean(v, axis=0) for k, v in kat.items()}
    ic = np.mean([np.mean([np.linalg.norm(v - ort[k]) for v in kat[k]])
                  for k in cfg.categories])
    ks = list(cfg.categories)
    dis = np.mean([np.linalg.norm(ort[ks[a]] - ort[ks[b]])
                   for a in range(3) for b in range(a + 1, 3)])
    return float(dis / (ic + 1e-9))


def elektrot_haritasi(resp, n=32):
    v = np.array([len(x) for x in resp.spike_times], dtype=float)
    if v.size == 0:
        return [0] * n
    return [1 if k.sum() > 0 else 0 for k in np.array_split(v, n)]


def piksel_haritasi(stim, n=64):
    y = np.array([len(s) for s in stim.spike_times], dtype=float)
    if y.size == 0:
        return [0.0] * n
    d = np.array([k.mean() if k.size else 0.0 for k in np.array_split(y, n)])
    if d.max() > 0:
        d = d / d.max()
    return [float(x) for x in d]


# ===================================================================
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pozitif", action="store_true",
                   help="kesin ogrenen durum — panel AYIRMALI")
    g.add_argument("--negatif", action="store_true",
                   help="kesin ogrenmeyen durum — panel AYIRMAMALI")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--no-panel", action="store_true")
    ap.add_argument("--esik-fark", type=float, default=8.0,
                    help="ayrisma sayilmasi icin gereken puan farki")
    a = ap.parse_args()

    pozitif = a.pozitif
    ad = "POZITIF (kesin ogrenen)" if pozitif else "NEGATIF (kesin ogrenmeyen)"

    print("=" * 66)
    print("  KALIBRASYON — %s" % ad)
    print("=" * 66)
    print("  Beklenen: panel cizgileri %s" % ("AYIRMALI" if pozitif else "AYIRMAMALI"))
    print()

    mon = None
    if not a.no_panel:
        from oi_monitor import Monitor
        mon = Monitor(port=a.port)
        mon.session(culture="kalibrasyon", note=ad)

    cfg = ExperimentConfig()
    rngA = np.random.default_rng(a.seed)
    rngB = np.random.default_rng(a.seed)

    # Kol A: sinanan durum   |   Kol B: her zaman ogrenmeyen kontrol
    orgA = KalibrasyonOrganoid(cfg, rng=rngA, ogrenir=pozitif)
    orgB = KalibrasyonOrganoid(cfg, rng=rngB, ogrenir=False)
    expA = ClosedLoopExperiment(orgA, cfg, rng=rngA)
    expB = ClosedLoopExperiment(orgB, cfg, rng=rngB)

    sepA0 = ayrilabilirlik(orgA, cfg)
    sepB0 = ayrilabilirlik(orgB, cfg)

    for i in range(a.trials):
        kat = cfg.categories[i % 3]
        sA = synthetic_stimulus(kat, DESEN[i % 3], cfg, rngA)
        sB = synthetic_stimulus(kat, DESEN[i % 3], cfg, rngB)
        if mon:
            mon.stimulus(kat, piksel_haritasi(sA))

        tA = expA.run_trial(sA, i)
        tB = expB.run_trial(sB, i)

        sepA = sepB = None
        if i % 15 == 0:
            sepA = ayrilabilirlik(orgA, cfg)
            sepB = ayrilabilirlik(orgB, cfg)

        if mon:
            for kol, t, sep in (("A", tA, sepA), ("B", tB, sepB)):
                resp = getattr(t, "response", None)
                mon.trial(arm=kol, trial=i, category=kat,
                          predicted=getattr(t, "predicted", None),
                          correct=bool(getattr(t, "correct", False)),
                          electrodes=elektrot_haritasi(resp) if resp is not None else None,
                          spikes=int(sum(len(x) for x in resp.spike_times)) if resp is not None else 0,
                          separability=sep)
        if a.delay:
            time.sleep(a.delay)

    accA = 100 * np.mean([t.correct for t in expA.trial_history[-40:]])
    accB = 100 * np.mean([t.correct for t in expB.trial_history[-40:]])
    sepA1 = ayrilabilirlik(orgA, cfg)
    sepB1 = ayrilabilirlik(orgB, cfg)
    fark = accA - accB
    sep_fark = (sepA1 - sepA0) - (sepB1 - sepB0)

    print()
    print("  SONUC")
    print("  " + "-" * 62)
    print("  kol A (sinanan)  dogruluk %.1f%%   ayrilabilirlik %.2f -> %.2f"
          % (accA, sepA0, sepA1))
    print("  kol B (kontrol)  dogruluk %.1f%%   ayrilabilirlik %.2f -> %.2f"
          % (accB, sepB0, sepB1))
    print("  fark             %+.1f puan        ayrilabilirlik net %+.2f"
          % (fark, sep_fark))
    print()

    ayirdi = abs(fark) >= a.esik_fark
    if pozitif:
        gecti = ayirdi and fark > 0 and sep_fark > 0.5
        print("  Panel ogrenmeyi %s" % ("GORDU" if gecti else "GOREMEDI"))
        if gecti:
            print("  -> KALIBRASYON GECTI. Simdi --negatif calistir.")
        else:
            print("  -> KALIBRASYON KALDI. Panel kor; ayar degistirilmeli.")
            print("     Denenecekler: sondalar/sonda_recurrent.py (tekrarlayan baglanti),")
            print("     ve uyaranin zamana yayilmasi.")
    else:
        gecti = not ayirdi
        print("  Panel yanlis alarm %s" % ("VERMEDI" if gecti else "VERDI"))
        if gecti:
            print("  -> KALIBRASYON GECTI. Iki testi de gecti; canli dokuya hazir.")
        else:
            print("  -> KALIBRASYON KALDI. Panel ogrenme yokken ayrisma gosteriyor.")

    print("=" * 66)

    if mon:
        mon.finish()

    return 0 if gecti else 1


if __name__ == "__main__":
    sys.exit(main())
