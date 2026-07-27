"""
tarama_32elektrot.py — 32 elektrotta calisan uyarilabilirlik ayarini bul.

Olculen 4 sey (hepsi ayni anda saglanmali):
  1. Organoid atesliyor mu        (bos trial orani dusuk)
  2. Doygunluga gitmiyor mu       (tum noronlar surekli atesler = ayrim yok)
  3. Kategoriler ayrisiyor mu     (between/within orani)
  4. STDP yalitilabiliyor mu      (lr acik vs kapali)
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
TEMEL = dict(n_electrodes=32, n_neurons=32, image_size=4, use_rank_order=True,
             stimulus_duration_s=0.5, reward_duration_s=0.2)


def hiz_vektoru(resp, n_neurons, sure):
    return np.array([len(x) for x in resp.spike_times], dtype=float) / sure


def olc(R, tau_syn, tohum=42, n_tekrar=6):
    """Bir ayar icin tum olcumleri dondur."""
    cfg = ExperimentConfig(r_membrane_mohm=R, tau_syn_ms=tau_syn, **TEMEL)
    rng = np.random.default_rng(tohum)
    org = SyntheticOrganoid(cfg, rng=rng)

    kat_vektorleri = {k: [] for k in cfg.categories}
    toplamlar = []
    for i in range(n_tekrar * 3):
        kat = cfg.categories[i % 3]
        s = synthetic_stimulus(kat, DESEN[i % 3], cfg, rng)
        resp = org.respond(s, timestamp=0.0)
        toplamlar.append(sum(len(x) for x in resp.spike_times))
        kat_vektorleri[kat].append(hiz_vektoru(resp, cfg.n_neurons, resp.duration_s))

    tp = np.array(toplamlar)
    bos_oran = float((tp == 0).mean())
    hz = tp.mean() / (cfg.n_neurons * cfg.stimulus_duration_s)
    # doygunluk: noron basina trial basina ortalama spike
    doygunluk = tp.mean() / cfg.n_neurons

    # kategori ayrimi
    ortalamalar = {k: np.mean(v, axis=0) for k, v in kat_vektorleri.items()}
    ic_yayilim = np.mean([
        np.mean([np.linalg.norm(v - ortalamalar[k]) for v in kat_vektorleri[k]])
        for k in cfg.categories
    ])
    ks = list(cfg.categories)
    dis_mesafe = np.mean([
        np.linalg.norm(ortalamalar[ks[a]] - ortalamalar[ks[b]])
        for a in range(3) for b in range(a + 1, 3)
    ])
    ayrim = float(dis_mesafe / (ic_yayilim + 1e-9))

    return dict(spike=float(tp.mean()), bos=bos_oran, hz=float(hz),
                doygunluk=float(doygunluk), ayrim=ayrim)


def stdp_yalitim(R, tau_syn, tohum=42, n_trial=20):
    """STDP acik vs kapali. Homeostatik kapali, yalitim icin."""
    def kol(lr):
        cfg = ExperimentConfig(r_membrane_mohm=R, tau_syn_ms=tau_syn,
                               learning_rate=lr, homeostatic=False, **TEMEL)
        r = np.random.default_rng(tohum)
        org = SyntheticOrganoid(cfg, rng=r)
        ex = ClosedLoopExperiment(org, cfg, rng=r)
        w0 = org.weights.copy()
        for i in range(n_trial):
            s = synthetic_stimulus(cfg.categories[i % 3], DESEN[i % 3], cfg, r)
            ex.run_trial(s, i)
        return float(np.abs(org.weights - w0).mean())
    return kol(0.1), kol(0.0)


print("=" * 78)
print("  32 ELEKTROT — UYARILABILIRLIK TARAMASI")
print("=" * 78)
print(f"  temel ayar: {TEMEL}")
print()
print(f"  {'R(MOhm)':>8}{'tau_syn':>9}{'spike/tr':>10}{'bos%':>7}{'Hz':>7}"
      f"{'spike/noron':>13}{'ayrim':>8}")
print("  " + "-" * 74)

adaylar = []
for R in [10, 15, 20, 25, 30, 40]:
    for tau in [5, 10, 20]:
        m = olc(R, tau)
        isaret = ""
        if m["bos"] <= 0.10 and 0.3 <= m["doygunluk"] <= 8.0 and m["ayrim"] >= 0.5:
            isaret = "  <-- aday"
            adaylar.append((R, tau, m))
        print(f"  {R:>8}{tau:>9}{m['spike']:>10.1f}{100*m['bos']:>6.0f}%"
              f"{m['hz']:>7.2f}{m['doygunluk']:>13.2f}{m['ayrim']:>8.2f}{isaret}")

print()
print("  Aday olcutu: bos<=%10, spike/noron 0.3-8 (doygun degil), ayrim>=0.5")
print()

if not adaylar:
    print("  ADAY YOK — olcutleri gevsetmek yerine baska bir knob gerekiyor.")
else:
    print("=" * 78)
    print("  ADAYLARDA STDP YALITIM TESTI")
    print("=" * 78)
    print(f"  {'R':>5}{'tau':>6}{'STDP acik':>14}{'STDP kapali':>14}{'sonuc':>18}")
    print("  " + "-" * 57)
    for R, tau, m in adaylar:
        a, b = stdp_yalitim(R, tau)
        ok = (a > 0 and b == 0.0)
        print(f"  {R:>5}{tau:>6}{a:>14.8f}{b:>14.8f}"
              f"{'YALITILIYOR' if ok else 'sorunlu':>18}")
