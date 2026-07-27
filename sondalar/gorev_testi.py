"""
gorev_testi.py — Gorev yeterince zor mu?

BULGU
-----
Okuma testi sunu gosterdi: STDP'nin hic calismadigi kolda bile meyveler
%83 dogrulukla ayirt ediliyor. Sans %33. Yani organoid daha egitim
baslamadan isin neredeyse tamamini yapmis durumda.

STDP'nin ogrenecek bir seyi kalmiyor. Aylardir aradigimiz cikmazin
sebebi bu: gorev fazla kolay.

COZUM ADAYI — ANAGRAM
---------------------
Ucu de ayni sembollerden olusan, sadece SIRASI farkli diziler:

    ART   . - . - . -
    RAT   . - . . - -
    TAR   - . - . - .

Ucunde de 6 sembol, 3 nokta, 3 cizgi. Toplam etkinlik ayni.
Sayarak ayirt edilemez — sirayi ogrenmek zorunda.

BU TESTIN SORUSU
----------------
Egitimsiz organoid bu dizileri kac dogrulukla ayiriyor?

    ~%33 (sans)  -> gorev dogru, STDP'ye is var
    %60+         -> bu da kolay, daha zorunu ariyoruz

Karsilastirma icin meyve gorevi de ayni kosulda olculur.

OLCUT — CALISTIRMADAN ONCE SABITLENDI
-------------------------------------
    taban dogruluk <= %45  -> gorev UYGUN
    %45 - %60             -> sinirda
    > %60                 -> hala kolay

KULLANIM
--------
    python sondalar/gorev_testi.py
    python sondalar/gorev_testi.py --tohum 15
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

from core.oi_gorev import (gorev_uyaran, gorev_etiketleri, gorev_config,
                           ANAGRAM, mors_uyaran)
from arama import TekrarlayanOrganoid
from okuma_testi import en_yakin_merkez_cv

# ---- olcutler: calistirmadan once sabitlendi ----
ESIK_UYGUN = 45.0
ESIK_SINIRDA = 60.0

def taban_dogruluk(gorev, cfg, tohum, n_tekrar=30, g=0.0, p=0.20, tekrar=1):
    """
    EGITIMSIZ organoidin taban dogrulugu.
    Hicbir agirlik guncellemesi yok — sadece yanit toplanip cevrimdisi cozulur.
    """
    cfg = gorev_config(gorev)
    rng = np.random.default_rng(tohum)
    org = TekrarlayanOrganoid(cfg, rng=rng, g_rec=g, p_rec=p)
    X, y = [], []
    etiketler = gorev_etiketleri(gorev, cfg)
    for i in range(n_tekrar * 3):
        k = etiketler[i % 3]
        s = gorev_uyaran(gorev, k, cfg, rng, i, tekrar)
        r = org.respond(s, timestamp=0.0)
        X.append([len(z) for z in r.spike_times])
        y.append(k)

    X = np.asarray(X, dtype=float)
    bos = int((X.sum(axis=1) == 0).sum())
    acc = en_yakin_merkez_cv(X, y, k=5, rng=np.random.default_rng(tohum + 1))
    return acc, bos, float(X.sum(axis=1).mean())


def hukum(acc):
    if acc <= ESIK_UYGUN:
        return "UYGUN"
    if acc <= ESIK_SINIRDA:
        return "sinirda"
    return "hala kolay"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tohum", type=int, default=10)
    ap.add_argument("--tekrar-sayisi", type=int, default=30)
    ap.add_argument("--g", type=float, default=0.0)
    ap.add_argument("--p", type=float, default=0.20)
    a = ap.parse_args()

    TABAN = [0, 1, 7, 42, 99, 123, 321, 777, 2024, 5555]
    tohumlar = (TABAN[:a.tohum] if a.tohum <= len(TABAN)
                else TABAN + list(range(10001, 10001 + a.tohum - len(TABAN))))

    cfg = gorev_config("meyve")

    print("=" * 74)
    print("  GOREV ZORLUGU TESTI")
    print("=" * 74)
    print("  Soru: EGITIMSIZ organoid gorevleri kac dogrulukla ayiriyor?")
    print("  Sans seviyesi: %33.3")
    print()
    print("  Olcutler (calistirmadan once sabitlendi):")
    print("    <= %.0f%%  -> gorev UYGUN, STDP'ye is var" % ESIK_UYGUN)
    print("    <= %.0f%%  -> sinirda" % ESIK_SINIRDA)
    print("    >  %.0f%%  -> hala kolay" % ESIK_SINIRDA)
    print()
    print("  %-8s%-14s%-10s%-14s%s" % ("tohum", "MEYVE", "bos/spike", "ANAGRAM", "bos/spike"))
    print("  " + "-" * 66)

    t0 = time.time()
    meyve, anagram = [], []
    for th in tohumlar:
        mA, mB, mS = taban_dogruluk("meyve", cfg, th, a.tekrar_sayisi, a.g, a.p)
        aA, aB, aS = taban_dogruluk("anagram", cfg, th, a.tekrar_sayisi, a.g, a.p)
        meyve.append(mA)
        anagram.append(aA)
        print("  %-8d%-14s%-10s%-14s%s" % (
            th, "%.1f%%" % mA, "%d/%.0f" % (mB, mS),
            "%.1f%%" % aA, "%d/%.0f" % (aB, aS)), flush=True)

    meyve, anagram = np.array(meyve), np.array(anagram)

    sat = []
    sat.append("=" * 74)
    sat.append("  GOREV ZORLUGU TESTI — SONUC")
    sat.append("  %s" % datetime.now().strftime("%Y%m%d-%H%M"))
    sat.append("=" * 74)
    sat.append("  %d tohum, gorev basina %d tekrar, EGITIMSIZ organoid"
               % (len(tohumlar), a.tekrar_sayisi * 3))
    sat.append("  Sans seviyesi: %33.3")
    sat.append("")
    sat.append("  MEYVE   taban dogruluk : %.1f%%  (std %.1f)   -> %s"
               % (meyve.mean(), meyve.std(), hukum(meyve.mean())))
    sat.append("  ANAGRAM taban dogruluk : %.1f%%  (std %.1f)   -> %s"
               % (anagram.mean(), anagram.std(), hukum(anagram.mean())))
    sat.append("")
    fark = meyve.mean() - anagram.mean()
    sat.append("  Anagram, meyveden %.1f puan daha zor." % fark)
    sat.append("")
    if anagram.mean() <= ESIK_UYGUN:
        sat.append("  SONUC: ANAGRAM GOREVI UYGUN.")
        sat.append("  Egitimsiz organoid sansa yakin. STDP'nin ogrenecegi bir sey var.")
        sat.append("  Yapilacak: gorevi anagram'a cevir, aramayi bu gorevle tekrarla.")
    elif anagram.mean() <= ESIK_SINIRDA:
        sat.append("  SONUC: SINIRDA.")
        sat.append("  Meyveden zor ama hala sansin belirgin uzerinde. Denenebilir,")
        sat.append("  ama daha uzun dizi veya daha fazla kategori gerekebilir.")
    else:
        sat.append("  SONUC: BU DA KOLAY.")
        sat.append("  Organoid zamanlama farkini da bedava ayirt ediyor. Daha zor bir")
        sat.append("  gorev gerekiyor — daha uzun diziler ya da daha cok kategori.")
    sat.append("=" * 74)
    metin = "\n".join(sat)
    print()
    print(metin)

    d = KOK / "sonuclar"
    d.mkdir(exist_ok=True)
    damga = datetime.now().strftime("%Y%m%d-%H%M")
    (d / ("gorev-%s.txt" % damga)).write_text(metin, encoding="utf-8")
    (d / ("gorev-%s.json" % damga)).write_text(json.dumps({
        "tur": "gorev_testi", "tohumlar": tohumlar,
        "tekrar_sayisi": a.tekrar_sayisi, "g_rec": a.g, "p_rec": a.p,
        "sure_dk": (time.time() - t0) / 60,
        "esikler": {"uygun": ESIK_UYGUN, "sinirda": ESIK_SINIRDA},
        "meyve": [float(x) for x in meyve],
        "anagram": [float(x) for x in anagram],
        "meyve_ort": float(meyve.mean()), "anagram_ort": float(anagram.mean()),
        "sonuc": hukum(anagram.mean()),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("  kaydedildi: sonuclar/gorev-%s.txt  ve  .json" % damga)


if __name__ == "__main__":
    main()
