"""
okuma_testi.py — Darbogaz okuma katmani mi?

SORU
----
Aramada su cikti: g=1.0 p=0.20 tekrar=6 ayarinda organoidin kategorileri
ayrisiyor (net +1.67, 10 tohumun 8'inde ayni yon) ama dogruluk kimildamiyor.

Iki acikama mumkun:
  (a) Ayrisma kategoriyle ilgili degil — bilgi aslinda yok.
  (b) Bilgi var ama online Hebbian okuyucu onu cikaramiyor. Darbogaz okuyucu.

Bu test ikisini ayirir.

YONTEM
------
Egitim bittikten sonra organoidin ham yanitlarini kaydediyoruz. Sonra bu
yanitlari CEVRIMDISI, basit ve saglam bir siniflandiriciyla cozuyoruz
(en yakin merkez, capraz dogrulamali).

Bu siniflandirici kapali dongunun parcasi DEGIL. Gorevi ogrenmek degil,
"bilgi orada mi" sorusuna cevap vermek. Bir tur ust sinir olcumu.

  cevrimdisi fark BUYUK, online fark SIFIR  -> darbogaz okuyucu (b)
  cevrimdisi fark da SIFIR                  -> bilgi yok (a)

OLCUT — CALISTIRMADAN ONCE SABITLENDI
-------------------------------------
Darbogazin okuyucu oldugu soylenebilmesi icin:
  (a) cevrimdisi dogruluk farki (STDP acik - kapali) >= +8.0 puan
  (b) tohumlarin en az %75'inde ayni isaret
  (c) ayni kosulda online fark < +8.0 puan  (yani online goremiyor)

Ucu birden saglanmazsa "darbogaz okuyucu" denmez.

KULLANIM
--------
    python sondalar/okuma_testi.py
    python sondalar/okuma_testi.py --tohum 20 --trials 300
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

KOK = Path(__file__).parent.parent.resolve()
for alt in ["", "core", "sim", "analysis", "sondalar"]:
    sys.path.insert(0, str(KOK / alt))

import numpy as np

from core.oi_loop import ClosedLoopExperiment
from core.oi_gorev import gorev_uyaran, gorev_etiketleri, gorev_config
from arama import TekrarlayanOrganoid

# ---- olcutler: calistirmadan once sabitlendi ----
ESIK_CEVRIMDISI = 8.0
ESIK_TOHUM_ORANI = 0.75
ESIK_ONLINE = 8.0


# ===================================================================
def en_yakin_merkez_cv(X, y, k=5, rng=None):
    """
    En yakin merkez siniflandirici, tabakali k-katli capraz dogrulama.
    Basit ve saglam — asiri uydurma riski dusuk.
    Doner: dogruluk yuzdesi.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    siniflar = np.unique(y)
    rng = rng or np.random.default_rng(0)

    # tabakali katlar
    katlar = [[] for _ in range(k)]
    for s in siniflar:
        idx = np.where(y == s)[0]
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            katlar[i % k].append(j)

    dogru = 0
    toplam = 0
    for i in range(k):
        test = np.array(katlar[i], dtype=int)
        egit = np.array([j for m in range(k) if m != i for j in katlar[m]], dtype=int)
        if len(test) == 0 or len(egit) == 0:
            continue
        merkez = {}
        for s in siniflar:
            se = egit[y[egit] == s]
            if len(se):
                merkez[s] = X[se].mean(axis=0)
        if len(merkez) < 2:
            continue
        etiketler = list(merkez.keys())
        M = np.stack([merkez[s] for s in etiketler])
        for j in test:
            d = np.linalg.norm(M - X[j], axis=1)
            if etiketler[int(np.argmin(d))] == y[j]:
                dogru += 1
            toplam += 1
    return 100.0 * dogru / max(toplam, 1)


def bir_kol(lr, g, p, tekrar, tohum, n_trial, kayit_son=90, gorev="anagram"):
    """
    Bir kolu calistirir. Doner:
      online_dogruluk, cevrimdisi_dogruluk
    Cevrimdisi olcum yalnizca son 'kayit_son' trial'in yanitlari uzerinden.
    """
    cfg = gorev_config(gorev, learning_rate=lr)
    rng = np.random.default_rng(tohum)
    org = TekrarlayanOrganoid(cfg, rng=rng, g_rec=g, p_rec=p)
    ex = ClosedLoopExperiment(org, cfg, rng=rng)

    X, y = [], []
    etiketler = gorev_etiketleri(gorev, cfg)
    basla = max(0, n_trial - kayit_son)
    for i in range(n_trial):
        kat = etiketler[i % 3]
        s = gorev_uyaran(gorev, kat, cfg, rng, i, tekrar)
        t = ex.run_trial(s, i)
        if i >= basla:
            resp = getattr(t, "response", None)
            if resp is not None:
                X.append([len(z) for z in resp.spike_times])
                y.append(kat)

    online = 100.0 * float(np.mean([t.correct for t in ex.trial_history[-40:]]))
    cevrimdisi = en_yakin_merkez_cv(X, y, k=5, rng=np.random.default_rng(tohum + 1))
    return online, cevrimdisi


# ===================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=0.20)
    ap.add_argument("--tekrar", type=int, default=6)
    ap.add_argument("--tohum", type=int, default=10)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--gorev", choices=["meyve", "anagram"], default="anagram")
    a = ap.parse_args()

    TABAN = [0, 1, 7, 42, 99, 123, 321, 777, 2024, 5555]
    if a.tohum <= len(TABAN):
        tohumlar = TABAN[:a.tohum]
    else:
        tohumlar = TABAN + list(range(10001, 10001 + a.tohum - len(TABAN)))

    print("=" * 74)
    print("  OKUMA KATMANI TESTI — gorev=%s  g=%.1f  p=%.2f  tekrar=%d"
          % (a.gorev.upper(), a.g, a.p, a.tekrar))
    print("=" * 74)
    print("  %d tohum x 2 kol x %d trial" % (len(tohumlar), a.trials))
    print("  Cevrimdisi cozucu: en yakin merkez, 5-katli capraz dogrulama")
    print("  (kapali dongunun parcasi degil — sadece 'bilgi orada mi' olcumu)")
    print()
    print("  %-8s%-12s%-12s%-12s%s" % ("tohum", "online A-B", "cevrimdisi A", "cevrimdisi B", "cevrimdisi A-B"))
    print("  " + "-" * 66)

    t0 = time.time()
    d_on, d_off, offA, offB = [], [], [], []
    for i, th in enumerate(tohumlar, 1):
        onA, cvA = bir_kol(0.01, a.g, a.p, a.tekrar, th, a.trials, gorev=a.gorev)
        onB, cvB = bir_kol(0.00, a.g, a.p, a.tekrar, th, a.trials, gorev=a.gorev)
        d_on.append(onA - onB)
        d_off.append(cvA - cvB)
        offA.append(cvA)
        offB.append(cvB)
        print("  %-8d%-12s%-12s%-12s%s   [%d/%d  %.0f dk]" % (
            th, "%+.1f" % (onA - onB), "%.1f%%" % cvA, "%.1f%%" % cvB,
            "%+.1f" % (cvA - cvB), i, len(tohumlar), (time.time() - t0) / 60), flush=True)

    d_on, d_off = np.array(d_on), np.array(d_off)
    tutarli = float(max((d_off > 0).mean(), (d_off < 0).mean()))

    gecti = (d_off.mean() >= ESIK_CEVRIMDISI
             and tutarli >= ESIK_TOHUM_ORANI
             and d_on.mean() < ESIK_ONLINE)

    sat = []
    sat.append("=" * 74)
    sat.append("  OKUMA KATMANI TESTI — SONUC")
    sat.append("  %s" % datetime.now().strftime("%Y%m%d-%H%M"))
    sat.append("=" * 74)
    sat.append("  ayar: g=%.1f  p=%.2f  tekrar=%d  |  %d tohum, %d trial"
               % (a.g, a.p, a.tekrar, len(tohumlar), a.trials))
    sat.append("")
    sat.append("  Olcutler (calistirmadan once sabitlendi):")
    sat.append("    cevrimdisi fark  >= %+.1f puan" % ESIK_CEVRIMDISI)
    sat.append("    isaret tutarlilik >= %.0f%%" % (100 * ESIK_TOHUM_ORANI))
    sat.append("    online fark      <  %+.1f puan" % ESIK_ONLINE)
    sat.append("")
    sat.append("  online fark      : %+.2f  (std %.2f)" % (d_on.mean(), d_on.std()))
    sat.append("  cevrimdisi A     : %.1f%%" % np.mean(offA))
    sat.append("  cevrimdisi B     : %.1f%%" % np.mean(offB))
    sat.append("  cevrimdisi fark  : %+.2f  (std %.2f)" % (d_off.mean(), d_off.std()))
    sat.append("  isaret tutarlilik: %.0f%%" % (100 * tutarli))
    sat.append("")
    if gecti:
        sat.append("  SONUC: DARBOGAZ OKUMA KATMANI.")
        sat.append("  Bilgi organoidin yanitinda var, online Hebbian okuyucu cikaramiyor.")
        sat.append("  Yapilacak: okuma katmani degistirilsin, doku degil.")
    elif d_off.mean() < 1.0:
        sat.append("  SONUC: BILGI YOK.")
        sat.append("  Cevrimdisi cozucu de fark bulamadi. Ayrisma artisi kategoriyle")
        sat.append("  ilgili degil. Okuma katmanini degistirmek ise yaramaz.")
    else:
        sat.append("  SONUC: KARARSIZ.")
        sat.append("  Cevrimdisi fark var ama olcutleri gecmiyor. Daha cok tohum gerekir.")
    sat.append("=" * 74)
    metin = "\n".join(sat)
    print()
    print(metin)

    d = KOK / "sonuclar"
    d.mkdir(exist_ok=True)
    damga = datetime.now().strftime("%Y%m%d-%H%M")
    (d / ("okuma-%s.txt" % damga)).write_text(metin, encoding="utf-8")
    (d / ("okuma-%s.json" % damga)).write_text(json.dumps({
        "tur": "okuma_testi", "gorev": a.gorev, "g_rec": a.g, "p_rec": a.p, "tekrar": a.tekrar,
        "tohumlar": tohumlar, "n_trial": a.trials,
        "sure_dk": (time.time() - t0) / 60,
        "esikler": {"cevrimdisi": ESIK_CEVRIMDISI,
                    "tohum_orani": ESIK_TOHUM_ORANI,
                    "online": ESIK_ONLINE},
        "online_farklari": [float(x) for x in d_on],
        "cevrimdisi_farklari": [float(x) for x in d_off],
        "cevrimdisi_A": [float(x) for x in offA],
        "cevrimdisi_B": [float(x) for x in offB],
        "sonuc": "darbogaz_okuyucu" if gecti else ("bilgi_yok" if d_off.mean() < 1.0 else "kararsiz"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("  kaydedildi: sonuclar/okuma-%s.txt  ve  .json" % damga)


if __name__ == "__main__":
    main()
