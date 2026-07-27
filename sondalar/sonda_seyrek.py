"""
sonda_seyrek.py — Seyrek/heterojen baglanti kanit denemesi.

SORU: Baglantiyi seyreklestirince STDP dogruluga fark yaratiyor mu?

Mevcut durum: her noron TUM elektrotlardan neredeyse ayni agirlikla
besleniyor (0.5 +- 0.05). Sonuc: butun noronlar ayni akimi goruyor,
hep birlikte atesliyorlar ya da hic ateslemiyorlar. Populasyon yaniti
tek boyutlu bir sayaca dusuyor (0 / ~35 / 100) ve agirliklar sonucu
degistirmiyor.

Bu sonda depodaki hicbir dosyayi degistirmez. SyntheticOrganoid'i
alt sinifla saran bir agirlik baslatma denemesidir.

Olculen:
  1. Populasyon cesitliligi  — hangi noronlar atesliyor, kategoriye gore degisiyor mu
  2. Ortusme (Jaccard)       — iki kategori ayni noronlari mi atesliyor
  3. STDP acik vs kapali     — ASIL SORU
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
for p in ["", "core", "sim", "analysis"]:
    sys.path.insert(0, str(ROOT / p))

import numpy as np
from core.oi_types import ExperimentConfig
from core.oi_signal import synthetic_stimulus
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid

DESEN = ["center", "edge", "stripe"]


class SeyrekOrganoid(SyntheticOrganoid):
    """
    Seyrek + heterojen baglanti.

    p      : bir noronun bir elektrota baglanma olasiligi (1.0 = mevcut davranis)
    Agirliklar uniform(0,1) — mevcut 0.5+-0.05 yerine gercek cesitlilik.
    Ortalama surus sabit tutulur: r_membrane p ile olceklenir.
    """
    def __init__(self, config=None, rng=None, p=1.0):
        super().__init__(config, rng=rng)
        self.p = p
        if p >= 1.0:
            return
        maske = self.rng.random((self.n_electrodes, self.n_neurons)) < p
        w = self.rng.random((self.n_electrodes, self.n_neurons))   # uniform(0,1)
        self.weights = np.where(maske, w, 0.0)


def yanit_deseni(org, cfg, rng, n_tekrar=5):
    """Kategori basina hangi noronlarin atesledigini topla."""
    kat = {k: [] for k in cfg.categories}
    for i in range(n_tekrar * 3):
        k = cfg.categories[i % 3]
        resp = org.respond(synthetic_stimulus(k, DESEN[i % 3], cfg, rng), timestamp=0.0)
        kat[k].append(np.array([len(x) for x in resp.spike_times]))
    return kat


def jaccard(a, b):
    A, B = set(np.where(a > 0)[0]), set(np.where(b > 0)[0])
    if not A and not B:
        return 1.0
    return len(A & B) / max(len(A | B), 1)


def dogruluk(p, lr, R, n_trial=100, tohum=7):
    cfg = ExperimentConfig(learning_rate=lr, r_membrane_mohm=R)
    rng = np.random.default_rng(tohum)
    org = SeyrekOrganoid(cfg, rng=rng, p=p)
    ex = ClosedLoopExperiment(org, cfg, rng=rng)
    for i in range(n_trial):
        ex.run_trial(synthetic_stimulus(cfg.categories[i % 3], DESEN[i % 3], cfg, rng), i)
    return 100 * np.mean([t.correct for t in ex.trial_history[-40:]])


print("=" * 76)
print("  BOLUM 1 — POPULASYON CESITLILIGI")
print("=" * 76)
print(f"  {'p':>5}{'R':>6}{'spike/trial':>14}{'atesleyen noron':>18}{'noronlar arasi std':>21}")
print("  " + "-" * 62)

for p, R in [(1.00, 10), (0.50, 20), (0.25, 40), (0.10, 100)]:
    cfg = ExperimentConfig(r_membrane_mohm=R)
    rng = np.random.default_rng(7)
    org = SeyrekOrganoid(cfg, rng=rng, p=p)
    kat = yanit_deseni(org, cfg, rng)
    hepsi = np.array([v for vs in kat.values() for v in vs])
    print(f"  {p:>5.2f}{R:>6}{hepsi.sum(axis=1).mean():>14.1f}"
          f"{(hepsi > 0).sum(axis=1).mean():>13.1f}/100{hepsi.std(axis=1).mean():>21.3f}")

print()
print("  noronlar arasi std = 0 demek: butun noronlar ayni sayida atesliyor")
print("  (yani populasyon tek boyutlu bir sayac, orunt bilgisi yok)")

print()
print("=" * 76)
print("  BOLUM 2 — KATEGORILER FARKLI NORONLARI MI ATESLIYOR (Jaccard ortusme)")
print("=" * 76)
print("  1.00 = ayni noronlar (ayirt edilemez) | 0.00 = tamamen farkli noronlar")
print()
print(f"  {'p':>5}{'banana-apple':>16}{'banana-pear':>15}{'apple-pear':>14}")
print("  " + "-" * 48)

for p, R in [(1.00, 10), (0.50, 20), (0.25, 40), (0.10, 100)]:
    cfg = ExperimentConfig(r_membrane_mohm=R)
    rng = np.random.default_rng(7)
    org = SeyrekOrganoid(cfg, rng=rng, p=p)
    kat = yanit_deseni(org, cfg, rng)
    ort = {k: np.mean(v, axis=0) for k, v in kat.items()}
    ks = list(cfg.categories)
    j = [jaccard(ort[ks[a]], ort[ks[b]]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    print(f"  {p:>5.2f}{j[0]:>16.2f}{j[1]:>15.2f}{j[2]:>14.2f}")

print()
print("=" * 76)
print("  BOLUM 3 — ASIL SORU: STDP DOGRULUGA FARK YARATIYOR MU")
print("=" * 76)
print("  (100 trial, son 40 trial dogrulugu, sans %33.3)")
print()
print(f"  {'p':>5}{'R':>6}{'STDP kapali':>15}{'STDP acik':>13}{'fark':>10}   sonuc")
print("  " + "-" * 66)

for p, R in [(1.00, 10), (0.50, 20), (0.25, 40), (0.10, 100)]:
    kapali = dogruluk(p, 0.0, R)
    acik = dogruluk(p, 0.01, R)
    fark = acik - kapali
    sonuc = "STDP ETKILI" if abs(fark) > 2.0 else "fark yok"
    print(f"  {p:>5.2f}{R:>6}{kapali:>14.1f}%{acik:>12.1f}%{fark:>+9.1f}   {sonuc}")

print()
print("  Karar: herhangi bir p'de anlamli fark varsa -> mimariyi duzgun yaz.")
print("         Hicbirinde yoksa -> sorun baglanti cesitliliginden daha derinde.")
