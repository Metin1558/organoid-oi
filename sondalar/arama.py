"""
arama.py — STDP'yi calistiran ayar var mi?

DURUM
-----
Panel kalibre edildi ve calisiyor (sondalar/kalibrasyon.py). Ogrenme gercekten
oldugunda goruyor, olmadiginda uydurmuyor. Yani olcu aleti saglam.

Geriye tek soru kaldi: LIF organoidi, Three-Factor STDP ile ogrenebilecegi
bir ayar var mi? Denetimde sekiz kosulda etki tam sifir cikti.

IKI ADAY
--------
1. TEKRARLAYAN BAGLANTI — su an noronlar birbirini hic gormuyor. Model
   tamamen ileri beslemeli. Gercek organoidde agin kendi ic dinamigi var.

2. ZAMANA YAYILMIS UYARAN — su an her elektrot tek bir spike atiyor, tek
   anlik akim tumsegi olusuyor. STDP zamanlamayla calisir; ona sirali bir
   hammadde vermiyoruz. Uyaran tekrarlanirsa organoid tek atis yerine bir
   dizi yanit verir.

OLCUT — CALISTIRMADAN ONCE SABITLENDI
-------------------------------------
Birincil olcut AYRILABILIRLIK'tir, dogruluk degil. Sebep: okuma katmani
dogrulugu telafi ediyor, gercek sinyali maskeliyor.

Bir ayarin "buldu" sayilmasi icin:
  (a) ayrilabilirlik net katkisi >= +0.50
  (b) tohumlarin en az %75'inde ayni isaret
  (c) dogruluk farki >= +8.0 puan

Ucu birden saglanmadikca bulundu denmez. Bu esikler burada yazilidir ve
sonuca gore degistirilmez.

KULLANIM
--------
    python sondalar/arama.py --tara
        Kaba tarama. Az tohum, kisa oturum. Aday ayarlari bulur.
        Suresi ~1 saat.

    python sondalar/arama.py --dogrula --g 1.0 --p 0.10 --tekrar 3
        Tek ayari cok tohumla, uzun oturumla dogrular.
        Suresi ~1 saat.

    python sondalar/arama.py --tara --hizli
        Cok kaba on bakis, ~10 dakika. Sadece calisiyor mu diye bakmak icin.

Sonuclar sonuclar/ klasorune yazilir. O dosyayi oldugu gibi paylasabilirsin.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

KOK = Path(__file__).parent.parent.resolve()
for alt in ["", "core", "sim", "analysis"]:
    sys.path.insert(0, str(KOK / alt))

import numpy as np

from core.oi_types import ExperimentConfig, OrganoidResponse, StimulusPattern
from core.oi_signal import synthetic_stimulus
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid, psp_kernel
from core.oi_gorev import (gorev_uyaran, gorev_etiketleri, gorev_config,
                           zamana_yay, DESEN)

# ---- olcutler: calistirmadan once sabitlendi, sonuca gore degistirilmez ----
ESIK_AYRILABILIRLIK = 0.50
ESIK_TOHUM_ORANI = 0.75
ESIK_DOGRULUK = 8.0


# ===================================================================
class TekrarlayanOrganoid(SyntheticOrganoid):
    """Ileri besleme + noronlar arasi seyrek, E/I dengeli tekrarlayan baglanti."""

    def __init__(self, config=None, rng=None, g_rec=0.0, p_rec=0.10, frac_inh=0.2):
        super().__init__(config, rng=rng)
        n = self.n_neurons
        self.g_rec = g_rec
        if g_rec <= 0:
            self.W_rec = None
            return
        inh = self.rng.random(n) < frac_inh
        maske = self.rng.random((n, n)) < p_rec
        np.fill_diagonal(maske, False)
        W = self.rng.random((n, n))
        W[inh, :] *= -4.0
        self.W_rec = np.where(maske, W, 0.0) * g_rec

    def _surus(self, stimulus, t):
        I = np.zeros((len(t), self.n_neurons))
        for i, spikes in enumerate(stimulus.spike_times):
            if len(spikes) == 0:
                continue
            w = self.weights[i]
            for ts in spikes:
                if ts >= stimulus.duration_s:
                    continue
                I += np.outer(psp_kernel(t, ts, tau_syn_ms=self.config.tau_syn_ms), w)
        return I

    def _sim(self, I_dis, dt_s):
        c = self.config
        n_t, n = I_dis.shape
        v = np.full(n, c.v_rest_mv, dtype=float)
        refrak = np.zeros(n, dtype=int)
        radim = max(1, int(c.refractory_ms / 1000.0 / dt_s))
        tau_s = c.tau_membrane_ms / 1000.0
        sonum = np.exp(-dt_s / (c.tau_syn_ms / 1000.0))
        I_rec = np.zeros(n)
        atesler = [[] for _ in range(n)]
        for k in range(n_t):
            I = I_dis[k] + I_rec
            aktif = refrak == 0
            dv = (-(v - c.v_rest_mv) + c.r_membrane_mohm * I) * (dt_s / tau_s)
            v = np.where(aktif, v + dv, c.v_reset_mv)
            refrak = np.where(aktif, refrak, refrak - 1)
            fired = aktif & (v >= c.v_thresh_mv)
            if fired.any():
                v[fired] = c.v_reset_mv
                refrak[fired] = radim
                for j in np.where(fired)[0]:
                    atesler[j].append(k * dt_s)
            if self.W_rec is not None:
                I_rec = I_rec * sonum + self.W_rec.T @ fired.astype(float)
        return [np.array(x) for x in atesler]

    def respond(self, stimulus, timestamp, post_stimulus_current=None, dt_ms=1.0):
        if self.W_rec is None:
            return super().respond(stimulus, timestamp, post_stimulus_current, dt_ms)
        dt_s = dt_ms / 1000.0
        t1 = np.arange(int(stimulus.duration_s / dt_s)) * dt_s
        I1 = self._surus(stimulus, t1)
        I1 += self.rng.normal(0, self._noise_current_na, I1.shape)
        z1 = self._sim(I1, dt_s)
        z2 = [np.array([]) for _ in range(self.n_neurons)]
        toplam = stimulus.duration_s
        if post_stimulus_current is not None and len(post_stimulus_current) > 0:
            I2 = np.tile(np.asarray(post_stimulus_current)[:, None], (1, self.n_neurons))
            I2 = I2 + self.rng.normal(0, self._noise_current_na, I2.shape)
            z2 = [x + stimulus.duration_s for x in self._sim(I2, dt_s)]
            toplam += len(post_stimulus_current) * dt_s
        return OrganoidResponse(
            spike_times=[np.concatenate([p, q]) for p, q in zip(z1, z2)],
            n_neurons=self.n_neurons, duration_s=toplam, timestamp=timestamp)


# ===================================================================
def ayrilabilirlik(org, cfg, tekrar, tohum=999, n=4, gorev="meyve"):
    rng = np.random.default_rng(tohum)
    etiketler = gorev_etiketleri(gorev, cfg)
    kat = {k: [] for k in etiketler}
    for i in range(n * 3):
        k = etiketler[i % 3]
        s = gorev_uyaran(gorev, k, cfg, rng, i, tekrar)
        r = org.respond(s, timestamp=0.0)
        kat[k].append(np.array([len(x) for x in r.spike_times], dtype=float))
    ort = {k: np.mean(v, axis=0) for k, v in kat.items()}
    ic = np.mean([np.mean([np.linalg.norm(v - ort[k]) for v in kat[k]])
                  for k in etiketler])
    ks = list(etiketler)
    dis = np.mean([np.linalg.norm(ort[ks[a]] - ort[ks[b]])
                   for a in range(3) for b in range(a + 1, 3)])
    return float(dis / (ic + 1e-9))


def bir_kol(lr, g, p, tekrar, tohum, n_trial, gorev="meyve"):
    cfg = gorev_config(gorev, learning_rate=lr)
    rng = np.random.default_rng(tohum)
    org = TekrarlayanOrganoid(cfg, rng=rng, g_rec=g, p_rec=p)
    ex = ClosedLoopExperiment(org, cfg, rng=rng)
    etiketler = gorev_etiketleri(gorev, cfg)
    sep0 = ayrilabilirlik(org, cfg, tekrar, gorev=gorev)
    for i in range(n_trial):
        s = gorev_uyaran(gorev, etiketler[i % 3], cfg, rng, i, tekrar)
        ex.run_trial(s, i)
    sep1 = ayrilabilirlik(org, cfg, tekrar, gorev=gorev)
    acc = 100 * float(np.mean([t.correct for t in ex.trial_history[-40:]]))
    return acc, sep1 - sep0


def bir_ayar(g, p, tekrar, tohumlar, n_trial, gorev="meyve"):
    """Bir ayari verilen tohumlarla esli kontrolle olcer."""
    d_acc, d_sep = [], []
    for th in tohumlar:
        aA, sA = bir_kol(0.01, g, p, tekrar, th, n_trial, gorev)
        aB, sB = bir_kol(0.00, g, p, tekrar, th, n_trial, gorev)
        d_acc.append(aA - aB)
        d_sep.append(sA - sB)
    d_acc, d_sep = np.array(d_acc), np.array(d_sep)
    if len(d_sep):
        oran = max((d_sep > 0).mean(), (d_sep < 0).mean())
    else:
        oran = 0.0
    return {
        "g_rec": g, "p_rec": p, "tekrar": tekrar,
        "n_tohum": len(tohumlar), "n_trial": n_trial,
        "dogruluk_farki_ort": float(d_acc.mean()),
        "dogruluk_farki_std": float(d_acc.std()),
        "ayrilabilirlik_net_ort": float(d_sep.mean()),
        "ayrilabilirlik_net_std": float(d_sep.std()),
        "isaret_tutarliligi": float(oran),
        "dogruluk_farklari": [float(x) for x in d_acc],
        "ayrilabilirlik_farklari": [float(x) for x in d_sep],
    }


def buldu_mu(r):
    return (r["ayrilabilirlik_net_ort"] >= ESIK_AYRILABILIRLIK
            and r["isaret_tutarliligi"] >= ESIK_TOHUM_ORANI
            and r["dogruluk_farki_ort"] >= ESIK_DOGRULUK)


# ===================================================================
def kaydet(ad, veri):
    d = KOK / "sonuclar"
    d.mkdir(exist_ok=True)
    damga = datetime.now().strftime("%Y%m%d-%H%M")
    j = d / ("%s-%s.json" % (ad, damga))
    t = d / ("%s-%s.txt" % (ad, damga))
    j.write_text(json.dumps(veri, indent=2, ensure_ascii=False), encoding="utf-8")

    sat = []
    sat.append("=" * 74)
    sat.append("  ARAMA SONUCU — %s" % ad)
    sat.append("  %s" % damga)
    sat.append("=" * 74)
    sat.append("  Olcutler (calistirmadan once sabitlendi):")
    sat.append("    ayrilabilirlik net katkisi >= %.2f" % ESIK_AYRILABILIRLIK)
    sat.append("    isaret tutarliligi         >= %.0f%%" % (100 * ESIK_TOHUM_ORANI))
    sat.append("    dogruluk farki             >= %.1f puan" % ESIK_DOGRULUK)
    sat.append("")
    sat.append("  %-7s%-7s%-8s%-11s%-11s%-9s%s" %
               ("g_rec", "p_rec", "tekrar", "ayrilab.", "dogruluk", "tutarli", "sonuc"))
    sat.append("  " + "-" * 70)
    for r in veri["sonuclar"]:
        sat.append("  %-7.1f%-7.2f%-8d%-11s%-11s%-9s%s" % (
            r["g_rec"], r["p_rec"], r["tekrar"],
            "%+.2f" % r["ayrilabilirlik_net_ort"],
            "%+.1f" % r["dogruluk_farki_ort"],
            "%.0f%%" % (100 * r["isaret_tutarliligi"]),
            "BULDU" if buldu_mu(r) else "-"))
    sat.append("")
    bulunan = [r for r in veri["sonuclar"] if buldu_mu(r)]
    if bulunan:
        sat.append("  %d aday olcutleri gecti." % len(bulunan))
        for r in bulunan:
            sat.append("    python sondalar/arama.py --dogrula --g %.1f --p %.2f --tekrar %d"
                       % (r["g_rec"], r["p_rec"], r["tekrar"]))
    else:
        sat.append("  Hicbir ayar olcutleri gecmedi.")
        en = max(veri["sonuclar"], key=lambda r: r["ayrilabilirlik_net_ort"])
        sat.append("  En yuksek ayrilabilirlik: %+.2f (g=%.1f p=%.2f tekrar=%d)"
                   % (en["ayrilabilirlik_net_ort"], en["g_rec"], en["p_rec"], en["tekrar"]))
    sat.append("=" * 74)
    metin = "\n".join(sat)
    t.write_text(metin, encoding="utf-8")
    print()
    print(metin)
    print()
    print("  kaydedildi: sonuclar/%s  ve  .json" % t.name)
    return t


# ===================================================================
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tara", action="store_true")
    g.add_argument("--dogrula", action="store_true")
    ap.add_argument("--hizli", action="store_true")
    ap.add_argument("--gorev", choices=["meyve", "anagram"], default="anagram",
                    help="varsayilan anagram (asil gorev)")
    ap.add_argument("--g", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--tekrar", type=int, default=1)
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--tohum", type=int, default=0,
                    help="dogrulamada kac tohum (varsayilan 10, ustu desteklenir)")
    a = ap.parse_args()

    t0 = time.time()

    if a.tara:
        if a.hizli:
            g_ler, p_ler, tek_ler = [0.0, 1.0], [0.10], [1, 3]
            tohumlar, n_trial = [7, 42], 60
            print("  KABA ON BAKIS — ~10 dakika")
        else:
            g_ler, p_ler, tek_ler = [0.0, 0.5, 1.0, 2.0], [0.05, 0.10, 0.20], [1, 3, 6]
            tohumlar, n_trial = [7, 42, 123], 120
            print("  TARAMA — ~1 saat")
        if a.trials:
            n_trial = a.trials
        toplam = len(g_ler) * len(p_ler) * len(tek_ler)
        print("  GOREV: %s" % a.gorev.upper())
        print("  %d ayar x %d tohum x 2 kol x %d trial" % (toplam, len(tohumlar), n_trial))
        print("  Birincil olcut: AYRILABILIRLIK (dogruluk degil)")
        print()
        print("  %-7s%-7s%-8s%-11s%-11s%-9s%s" %
              ("g_rec", "p_rec", "tekrar", "ayrilab.", "dogruluk", "tutarli", "sonuc"))
        print("  " + "-" * 70)

        sonuclar = []
        say = 0
        for gg in g_ler:
            for pp in p_ler:
                if gg == 0.0 and pp != p_ler[0]:
                    continue  # g=0 iken p'nin anlami yok
                for tt in tek_ler:
                    say += 1
                    r = bir_ayar(gg, pp, tt, tohumlar, n_trial, a.gorev)
                    sonuclar.append(r)
                    print("  %-7.1f%-7.2f%-8d%-11s%-11s%-9s%s   [%d/%d  %.0f dk]" % (
                        gg, pp, tt,
                        "%+.2f" % r["ayrilabilirlik_net_ort"],
                        "%+.1f" % r["dogruluk_farki_ort"],
                        "%.0f%%" % (100 * r["isaret_tutarliligi"]),
                        "BULDU" if buldu_mu(r) else "-",
                        say, toplam, (time.time() - t0) / 60), flush=True)

        kaydet("tarama", {
            "tur": "tarama", "gorev": a.gorev, "hizli": a.hizli, "tohumlar": tohumlar,
            "n_trial": n_trial, "sure_dk": (time.time() - t0) / 60,
            "esikler": {"ayrilabilirlik": ESIK_AYRILABILIRLIK,
                        "tohum_orani": ESIK_TOHUM_ORANI,
                        "dogruluk": ESIK_DOGRULUK},
            "sonuclar": sonuclar})

    else:
        TABAN = [0, 1, 7, 42, 99, 123, 321, 777, 2024, 5555]
        n_toh = a.tohum or len(TABAN)
        if n_toh <= len(TABAN):
            tohumlar = TABAN[:n_toh]
        else:
            # 10'un uzerinde istenirse deterministik olarak uzat
            ek = list(range(10001, 10001 + (n_toh - len(TABAN))))
            tohumlar = TABAN + ek
        n_trial = a.trials or 300
        print("  DOGRULAMA — gorev=%s  g=%.1f  p=%.2f  tekrar=%d"
              % (a.gorev.upper(), a.g, a.p, a.tekrar))
        print("  %d tohum x 2 kol x %d trial" % (len(tohumlar), n_trial))
        print()
        d_acc, d_sep = [], []
        for i, th in enumerate(tohumlar, 1):
            aA, sA = bir_kol(0.01, a.g, a.p, a.tekrar, th, n_trial, a.gorev)
            aB, sB = bir_kol(0.00, a.g, a.p, a.tekrar, th, n_trial, a.gorev)
            d_acc.append(aA - aB)
            d_sep.append(sA - sB)
            print("  tohum %5d: dogruluk %+6.1f   ayrilabilirlik %+6.2f   [%d/%d  %.0f dk]"
                  % (th, aA - aB, sA - sB, i, len(tohumlar), (time.time() - t0) / 60),
                  flush=True)
        r = {"g_rec": a.g, "p_rec": a.p, "tekrar": a.tekrar,
             "n_tohum": len(tohumlar), "n_trial": n_trial,
             "dogruluk_farki_ort": float(np.mean(d_acc)),
             "dogruluk_farki_std": float(np.std(d_acc)),
             "ayrilabilirlik_net_ort": float(np.mean(d_sep)),
             "ayrilabilirlik_net_std": float(np.std(d_sep)),
             "isaret_tutarliligi": float(max((np.array(d_sep) > 0).mean(),
                                             (np.array(d_sep) < 0).mean())),
             "dogruluk_farklari": [float(x) for x in d_acc],
             "ayrilabilirlik_farklari": [float(x) for x in d_sep]}
        kaydet("dogrulama", {
            "tur": "dogrulama", "gorev": a.gorev, "sure_dk": (time.time() - t0) / 60,
            "esikler": {"ayrilabilirlik": ESIK_AYRILABILIRLIK,
                        "tohum_orani": ESIK_TOHUM_ORANI,
                        "dogruluk": ESIK_DOGRULUK},
            "sonuclar": [r]})


if __name__ == "__main__":
    main()
