"""
oi_gorev.py — Gorev tanimlari.

Sistem iki gorevi birden tasir:

MEYVE (kontrol gorevi)
    Uc gorsel kategori: banana, apple, pear.
    Egitimsiz organoid bunu %100 dogrulukla ayirir — cunku uc kategori
    farkli SAYIDA noron uyandirir (0 / ~35 / 100). Ayirt edilecek bir
    orgu yok, sayilacak bir sayi var. STDP'ye is kalmaz.

    Artik asil gorev degil, KONTROL gorevidir: "ayni sistem meyvede
    ogrenmedi, anagramda ogrendi" karsilastirmasini mumkun kilar.

ANAGRAM (asil gorev)
    Uc mors dizisi. Ucu de ayni sembollerden olusur, sadece SIRASI farkli:

        art   . - . - . -
        rat   . - . . - -
        tar   - . - . - .

    Ucunde de 6 sembol, 3 nokta, 3 cizgi. Toplam etkinlik ayni oldugu
    icin sayarak ayirt edilemez. Egitimsiz organoid %35.7 yapar
    (sans %33.3, 10 tohum, std 5.2). Yani ogrenmek zorunda.

    Elektrot kimligi de kategoriyi ele vermez: her elektrot her sembolde
    ates eder. Bilgi yalnizca ZAMANLAMADADIR — STDP'nin calistigi eksende.

KULLANIM
--------
    from core.oi_gorev import gorev_uyaran, gorev_etiketleri

    for i in range(n):
        etiket = gorev_etiketleri("anagram", cfg)[i % 3]
        stim = gorev_uyaran("anagram", etiket, cfg, rng, i)
"""

import numpy as np

from core.oi_types import ExperimentConfig, StimulusPattern
from core.oi_signal import synthetic_stimulus

DESEN = ["center", "edge", "stripe"]

# ucu de 3 nokta + 3 cizgi — toplam etkinlik ayni, yalnizca sira farkli
ANAGRAM = {
    "art": [0, 1, 0, 1, 0, 1],
    "rat": [0, 1, 0, 0, 1, 1],
    "tar": [1, 0, 1, 0, 1, 0],
}

GOREVLER = ("meyve", "anagram")


# -------------------------------------------------------------------
def zamana_yay(stim, tekrar, cfg):
    """
    Uyarani zamana yayar: ayni orgunun pencere icinde 'tekrar' kez
    yinelenmesi. tekrar=1 mevcut davranistir.

    Meyve gorevinde kullanilir. Anagram zaten zamansaldir.
    """
    if tekrar <= 1:
        return stim
    sure = cfg.stimulus_duration_s
    adim = sure / tekrar
    yeni = []
    for tr in stim.spike_times:
        if len(tr) == 0:
            yeni.append(np.array([]))
            continue
        taban = np.asarray(tr)
        taban = taban - taban.min() if taban.size else taban
        olcek = adim / max(taban.max(), 1e-9) * 0.6 if taban.max() > 0 else 0.0
        yeni.append(np.sort(np.concatenate(
            [taban * olcek + k * adim for k in range(tekrar)])))
    return StimulusPattern(spike_times=yeni, n_electrodes=stim.n_electrodes,
                           duration_s=sure, label=stim.label)


# -------------------------------------------------------------------
def mors_uyaran(kelime, cfg, rng, nokta_spike=1, cizgi_spike=3,
                elektrot_orani=0.25, jitter_s=0.002):
    """
    Mors dizisini uyarana cevirir.

    nokta_spike / cizgi_spike : sembol basina spike sayisi. Toplam surusu
        dusurmek icin kucultulebilir (anagram organoidi sert suruyor —
        trial basina ~2400 spike, meyvede ~45).

    elektrot_orani : kac elektrodun sinyali tasidigi. Hangi elektrotlarin
        secildigi kategoriden BAGIMSIZDIR, yani kimlik sizintisi olmaz.
        Yalnizca toplam surusu azaltmaya yarar.

        Olculen (4 tohum, egitimsiz organoid):
            %100 -> 48.7 Hz/noron, taban dogruluk %36.1
            %50  -> 24.9 Hz/noron, taban dogruluk %35.0
            %25  ->  8.5 Hz/noron, taban dogruluk %41.1   <- varsayilan

        Varsayilan %25 secildi: gercek doku 0.4-2 Hz ateslerken 48 Hz
        savunulamaz, ayrica STDP hesabi spike sayisiyla karesel buyudugu
        icin tarama suresi 6 kat kisaliyor. Taban dogruluk hala %45
        esiginin altinda.
    """
    sembol = ANAGRAM[kelime]
    sure = cfg.stimulus_duration_s
    dilim = sure / len(sembol)
    aktif = dilim * 0.55

    n_el = cfg.n_electrodes
    if elektrot_orani >= 1.0:
        tasiyan = set(range(n_el))
    else:
        # kategoriden bagimsiz, sabit alt kume
        k = max(1, int(round(n_el * elektrot_orani)))
        tasiyan = set(np.linspace(0, n_el - 1, k).astype(int).tolist())

    izler = []
    for e in range(n_el):
        if e not in tasiyan:
            izler.append(np.array([]))
            continue
        t = []
        for k, s in enumerate(sembol):
            bas = k * dilim
            n_sp = cizgi_spike if s else nokta_spike
            if n_sp == 1:
                t.append(bas + aktif * 0.15 + rng.normal(0, jitter_s))
            else:
                for j in range(n_sp):
                    t.append(bas + aktif * (0.1 + 0.8 * j / (n_sp - 1))
                             + rng.normal(0, jitter_s))
        izler.append(np.sort(np.clip(np.array(t), 0.0, sure - 1e-4)))

    return StimulusPattern(spike_times=izler, n_electrodes=n_el,
                           duration_s=sure, label=kelime)


# -------------------------------------------------------------------
def gorev_etiketleri(gorev, cfg):
    """Gorevin uc kategori etiketi."""
    if gorev == "anagram":
        return list(ANAGRAM.keys())
    return list(cfg.categories)


def gorev_uyaran(gorev, etiket, cfg, rng, indeks=0, tekrar=1, **kw):
    """
    Gorev ne olursa olsun tek arayuz.

    gorev   : "meyve" veya "anagram"
    etiket  : gorev_etiketleri() listesinden bir eleman
    indeks  : trial numarasi (meyvede gorsel desen secimi icin)
    tekrar  : meyve gorevinde zamana yayma katsayisi
    """
    if gorev == "anagram":
        return mors_uyaran(etiket, cfg, rng, **kw)
    if gorev == "meyve":
        return zamana_yay(
            synthetic_stimulus(etiket, DESEN[indeks % 3], cfg, rng), tekrar, cfg)
    raise ValueError("bilinmeyen gorev: %r (secenekler: %s)" % (gorev, GOREVLER))


def gorev_config(gorev, **kw):
    """
    Gorevin etiketleriyle uyumlu bir ExperimentConfig uretir.

    Okuma katmani config.categories listesine kilitlidir; anagram
    calistirilirken bu liste ["art","rat","tar"] olmak zorundadir.
    """
    if gorev == "anagram":
        kw.setdefault("categories", list(ANAGRAM.keys()))
    elif gorev != "meyve":
        raise ValueError("bilinmeyen gorev: %r" % gorev)
    return ExperimentConfig(**kw)
