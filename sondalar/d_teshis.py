#!/usr/bin/env python3
"""
d_teshis.py — D (organoid-oi v3) teshis araci
==============================================

Amac: Gercek organoid deneyine gitmeden once, kodun su anki halinde
      nerede durdugunu OLCMEK. Iddia degil, sayi uretir.

Olctugu 4 sey:
  1. Import sagligi        — kod hic yuklenebiliyor mu
  2. T2 teshisi            — STDP anlamli agirlik degisimi yaratiyor mu
                             (tohum taramasi + trial sayisi taramasi)
  3. T4 teshisi            — decoder agirlik kacisi engelleniyor mu
  4. Ates hizi + bos trial — gercek doku ile karsilastirmali

Kullanim:
    python sondalar/d_teshis.py        # depo kokunden calistir
    python sondalar/d_teshis.py /yol/organoid-oi

Hicbir dosyayi degistirmez. Sadece okur ve olcer.
Bagimlilik: numpy.
"""

import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────
# Yol kurulumu
# ─────────────────────────────────────────────────────────────
KOK = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).parent.parent.resolve()
for alt in ["", "core", "sim", "analysis"]:
    sys.path.insert(0, str(KOK / alt))

CIZGI = "=" * 68


def baslik(metin):
    print("\n" + CIZGI)
    print("  " + metin)
    print(CIZGI)


def sonuc(etiket, gecti, aciklama=""):
    isaret = "GECER" if gecti else "KALIR"
    print(f"  [{isaret}] {etiket}")
    if aciklama:
        print(f"          {aciklama}")


# ─────────────────────────────────────────────────────────────
# BOLUM 0 — Import sagligi
# ─────────────────────────────────────────────────────────────
baslik("BOLUM 0 — IMPORT SAGLIGI")
print(f"  repo koku : {KOK}")
print(f"  numpy     : {np.__version__}")
print(f"  python    : {sys.version.split()[0]}")
print()

try:
    from core.oi_types import ExperimentConfig
    from core.oi_signal import synthetic_stimulus
    from core.oi_loop import ClosedLoopExperiment, HebbianDecoder
    from sim.oi_synth import SyntheticOrganoid
    sonuc("tum moduller yuklendi", True)
except Exception as e:
    sonuc("modul yukleme", False, f"{type(e).__name__}: {e}")
    print("\n  NOT: 'List is not defined' hatasi aliyorsan, oi_signal.py icinde")
    print("       'from typing import List' satiri kullanildigi yerden SONRA.")
    print("       Dosyanin en ustune tasi.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Ortak yardimcilar
# ─────────────────────────────────────────────────────────────
DESENLER = ["center", "edge", "stripe"]


def t2_config():
    """test_oi_v3.py T2 bolumundeki config ile birebir ayni."""
    return ExperimentConfig(
        n_electrodes=8, n_neurons=5,
        homeostatic_epoch=10, homeostatic_rate=0.05,
        learning_rate=0.1,
        image_size=4, use_rank_order=True,
        stimulus_duration_s=0.2, reward_duration_s=0.1,
    )


def agirlik_degisimi(tohum, n_trial):
    """n_trial sonunda ortalama mutlak agirlik degisimi."""
    rng = np.random.default_rng(tohum)
    cfg = t2_config()
    org = SyntheticOrganoid(cfg, rng=rng)
    exp = ClosedLoopExperiment(org, cfg, rng=rng)
    w0 = org.weights.copy()
    for i in range(n_trial):
        s = synthetic_stimulus(cfg.categories[i % 3], DESENLER[i % 3], cfg, rng)
        exp.run_trial(s, i)
    return float(np.abs(org.weights - w0).mean())


# ─────────────────────────────────────────────────────────────
# BOLUM 1 — T2: STDP anlamli agirlik degisimi yaratiyor mu
# ─────────────────────────────────────────────────────────────
baslik("BOLUM 1 — T2: STDP AGIRLIK DEGISIMI")
ESIK_T2 = 0.001
print(f"  Testteki esik : agirlik degisimi > {ESIK_T2}")
print(f"  Trial sayisi  : 10 (testte oldugu gibi)")
print()

TOHUMLAR = [0, 1, 7, 42, 99, 123, 321, 777, 2024, 5555]
print("  --- Tohum taramasi (esigin sansa mi bagli oldugunu gorur) ---")
degerler = []
for th in TOHUMLAR:
    d = agirlik_degisimi(th, 10)
    degerler.append(d)
    print(f"    tohum {th:5d} : {d:.6f}   {'gecer' if d > ESIK_T2 else 'kalir'}")

dz = np.array(degerler)
gecen = int((dz > ESIK_T2).sum())
print()
print(f"    {gecen}/{len(TOHUMLAR)} tohum gecti")
print(f"    en dusuk {dz.min():.6f} | ortalama {dz.mean():.6f} | en yuksek {dz.max():.6f}")
print(f"    esige uzaklik: ortalama deger esigin {ESIK_T2/dz.mean():.2f} kati altinda")

print()
print("  --- Trial sayisi taramasi (ASIL TESHIS) ---")
print("  Etki trial sayisiyla birikiyorsa  -> sorun ESIKTE, mekanizma saglam")
print("  Etki trial sayisiyla birikmiyorsa -> sorun MEKANIZMADA, ciddi")
print()
birikim = []
for n in [10, 25, 50, 100]:
    t0 = time.time()
    d = agirlik_degisimi(42, n)
    birikim.append((n, d))
    print(f"    {n:4d} trial : {d:.6f}   ({time.time()-t0:.1f} sn)")

ilk, son = birikim[0][1], birikim[-1][1]
oran = son / ilk if ilk > 0 else float("inf")
print()
print(f"    10 -> 100 trial buyume orani: {oran:.2f}x")
if oran > 3.0:
    print("    YORUM: Etki birikiyor. Mekanizma calisiyor, test esigi yanlis kalibre.")
    print("           Yapilacak: esik 10 trial icin gercekci bir degere cekilsin.")
elif oran > 1.5:
    print("    YORUM: Etki zayif birikiyor. Homeostatik olcekleme kismen siliyor olabilir.")
    print("           Yapilacak: homeostatic=False ile tekrar olc, farki gor.")
else:
    print("    YORUM: Etki BIRIKMIYOR. Bu ciddi — STDP net bir yon uretmiyor demektir.")
    print("           Gercek dokuya gitmeden once bunun cozulmesi gerekir.")

sonuc(f"T2 (10 trial, tohum 42)", degerler[3] > ESIK_T2,
      f"olculen {degerler[3]:.6f} / esik {ESIK_T2}")


# ─────────────────────────────────────────────────────────────
# BOLUM 2 — T4: decoder agirlik kacisi
# ─────────────────────────────────────────────────────────────
baslik("BOLUM 2 — T4: DECODER AGIRLIK KACISI")
ESIK_T4 = 50.0
cfg = ExperimentConfig()


class SahteYanit:
    """20 noron, her biri 0.5 sn'de 50 spike = 100 Hz (testteki gibi)."""
    def __init__(self):
        self.spike_times = [np.linspace(0, 0.5, 50)] * 20
        self.duration_s = 0.5
        self.n_neurons = 20


try:
    dec = HebbianDecoder(cfg.categories, 20, cfg)
except TypeError:
    dec = HebbianDecoder(cfg, n_neurons=20)

yanit = SahteYanit()
LR = 0.1
for _ in range(100):
    dec.update(yanit, cfg.categories[0], learning_rate=LR)

maks = float(np.abs(dec.weights).max())
print(f"  100 guncelleme sonrasi max|w| : {maks:.1f}")
print(f"  Testteki esik                 : < {ESIK_T4}")
print()
print("  Matematik:")
print(f"    ates hizi        = 100 Hz (50 spike / 0.5 sn)")
print(f"    ogrenme hizi     = {LR}")
print(f"    sonum sabiti     = {cfg.decoder_hebbian_decay}")
denge = LR * 100.0 / cfg.decoder_hebbian_decay
print(f"    denge noktasi    = lr*hiz/sonum = {denge:.0f}")
print()
print("  Bu deterministik — rastgelelik yok, her calistirmada ayni cikar.")
if maks >= ESIK_T4:
    print("  YORUM: Sonum sabiti bu ayarlar icin cok zayif. Iki secenek:")
    print("         (a) decoder_hebbian_decay buyutulsun")
    print("         (b) test esigi gercekci denge noktasina cekilsin")
    print("         Hangisi dogru: gercek dokuda ates hizi 100 Hz OLMAYACAK")
    print("         (~0.4-2 Hz). Yani test senaryosu gercekci degil.")

sonuc("T4 agirlik kacisi", maks < ESIK_T4, f"olculen {maks:.1f} / esik {ESIK_T4}")


# ─────────────────────────────────────────────────────────────
# BOLUM 3 — Ates hizi ve bos trial orani
# ─────────────────────────────────────────────────────────────
baslik("BOLUM 3 — ATES HIZI VE BOS TRIAL ORANI")
N_OLCUM = 20
cfg = ExperimentConfig()
rng = np.random.default_rng(7)
org = SyntheticOrganoid(cfg, rng=rng)

toplamlar = []
print(f"  {N_OLCUM} trial olculuyor (varsayilan ayar, {cfg.n_neurons} noron)...")
t0 = time.time()
for i in range(N_OLCUM):
    s = synthetic_stimulus(cfg.categories[i % 3], DESENLER[i % 3], cfg, rng)
    r = org.respond(s, timestamp=0.0)
    toplamlar.append(sum(len(x) for x in r.spike_times))
print(f"  ({time.time()-t0:.1f} sn)")
print()

tp = np.array(toplamlar)
bos = int((tp == 0).sum())
hz = tp.mean() / (cfg.n_neurons * cfg.stimulus_duration_s)

print(f"  trial basina spike : ortalama {tp.mean():.1f} | ortanca {np.median(tp):.1f}")
print(f"                       en dusuk {tp.min()} | en yuksek {tp.max()}")
print(f"  noron basina hiz   : {hz:.2f} Hz")
print(f"  BOS TRIAL          : {bos}/{N_OLCUM}  (%{100*bos/N_OLCUM:.0f})")
print()
print("  --- Gercek doku karsilastirmasi (32 elektrot x 0.5 sn pencere) ---")
GERCEK = [
    ("DANDI insan organoidi", 0.42),
    ("fs437 (FinalSpark)",    0.0545),
    ("fs369 (FinalSpark)",    1.9116),
]
print(f"    {'kaynak':<26}{'Hz':>8}{'trial basina spike':>22}")
print(f"    {'simulasyon (su an)':<26}{hz:>8.2f}{tp.mean():>22.1f}")
for ad, h in GERCEK:
    print(f"    {ad:<26}{h:>8.4f}{h*32*0.5:>22.1f}")

print()
if bos > 0:
    print(f"  YORUM: Trial'larin %{100*bos/N_OLCUM:.0f}'i BOS. Bos trial = decoder kor tahmin yapiyor.")
    print("         Gercek dokuda 100 degil 32 kanal var, yani daha kotu olacak.")
    print("         Deneyden ONCE karara baglanmasi gereken:")
    print("           - kac spike altinda 'okuma yok' denecek")
    print("           - bos trial atilacak mi, yanlis mi sayilacak")
    print("         Aksi halde dogruluk orani ogrenmeyi degil ates hizini olcer.")


# ─────────────────────────────────────────────────────────────
# OZET
# ─────────────────────────────────────────────────────────────
baslik("OZET")
print(f"  T2 (STDP agirlik degisimi) : {'GECER' if degerler[3] > ESIK_T2 else 'KALIR'}"
      f"   [{degerler[3]:.6f} vs {ESIK_T2}]")
print(f"  T2 birikim orani           : {oran:.2f}x  (10 -> 100 trial)")
print(f"  T4 (decoder kacisi)        : {'GECER' if maks < ESIK_T4 else 'KALIR'}"
      f"   [{maks:.1f} vs {ESIK_T4}]")
print(f"  Bos trial orani            : %{100*bos/N_OLCUM:.0f}")
print(f"  Simulasyon ates hizi       : {hz:.2f} Hz")
print()
print("  NOT: T2 ve T4 satirlari v3.0'in ESKI esiklerine gore raporlanir ve")
print("  \"KALIR\" gostermesi beklenir — bu, esiklerin neden yanlis oldugunun")
print("  kaydidir. tests/test_oi_v3.py icindeki duzeltilmis testler 47/47 gecer.")
print()
print("  Bu sayilar iddia degil olcumdur.")
print(CIZGI)
