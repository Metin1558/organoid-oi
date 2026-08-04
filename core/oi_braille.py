"""
oi_braille.py — Gorev tanimi: Braille (kabartma yazi).

FIKIR
-----
Onceki iki gorev denemesi (meyve, mors-anagram) hep ayni hataya dustu:
kategoriler arasindaki fark, TOPLAM sinyal miktarindan geliyordu. STDP
zamanlamaya bakar, miktara degil — o yuzden miktar farki onu hicbir zaman
gercekten calistirmadi.

Braille bunu yapisal olarak imkansiz kilar. Bir Braille hucresi 6 sabit
noktadir (2 sutun x 3 satir). Her harf, bu 6 noktadan HANGILERININ kabarik
oldugunu belirler — kac tanesinin degil. Ayni sayida kabarik noktaya sahip
harfler var (orn. G ve V, ikisi de 4 nokta) ve bu harfler arasindaki fark
SADECE hangi noktalarin kabarik oldugudur.

    G = 1,2,4,5     . .
                     . .
                    (bos, bos)

    V = 1,2,3,6     . .
                     . .
                    (bos, .)

Organoide gosterirken, 6 noktayi KOR BIR PARMAGIN Braille'i tarama
hareketini taklit ederek zamana yayiyoruz: nokta 1 once, nokta 6 en son,
sabit sirayla. Kabarik nokta o zaman diliminde bir darbe demek, duz nokta
o dilimde tam sessizlik demek. Boylece pozisyon bilgisi, STDP'nin zaten
calistigi eksene (zamanlama) tasinmis olur.

ELEKTROT ESLEMESI
------------------
Varsayilan olarak n_electrodes=6: elektrot i, dogrudan Braille noktasi
i+1'i temsil eder. Daha fazla elektrot istenirse (orn. donanimda 32 varsa),
elektrotlar 6 noktaya DONGUSEL olarak gruplanir (elektrot 0 ve 6, ikisi de
nokta 1'i tasir) — boylece ayni tasarim daha fazla elektrotla da calisir.

ZORLUK SIRALAMASI
------------------
4 noktali 9 harf var (G,N,P,R,T,V,W,X,Z). En yakin cift P/R (3 ortak nokta,
1 fark) — en uzak cift G/V (2 ortak nokta, 2 fark). Once en kolayla
(G/V) basliyoruz; calisirsa kademeli olarak zorlastiririz.

KULLANIM
--------
    from core.oi_braille import braille_uyaran, braille_etiketleri, braille_config

    cfg = braille_config()
    for i in range(n):
        harf = braille_etiketleri(cfg)[i % 2]
        stim = braille_uyaran(harf, cfg, rng, tekrar=1)
"""

import numpy as np

from core.oi_types import ExperimentConfig, StimulusPattern

# Standart Ingilizce Braille noktalari (1-6 numarali, 2 sutun x 3 satir).
# 1=kabarik, 0=duz. Tam alfabe (ileride zorluk kademesi icin) burada
# tutuluyor; su an sadece 'g' ve 'v' kullaniliyor.
BRAILLE_TAM = {
    "a": [1,0,0,0,0,0], "b": [1,1,0,0,0,0], "c": [1,0,0,1,0,0],
    "d": [1,0,0,1,1,0], "e": [1,0,0,0,1,0], "f": [1,1,0,1,0,0],
    "g": [1,1,0,1,1,0], "h": [1,1,0,0,1,0], "i": [0,1,0,1,0,0],
    "j": [0,1,0,1,1,0], "k": [1,0,1,0,0,0], "l": [1,1,1,0,0,0],
    "m": [1,0,1,1,0,0], "n": [1,0,1,1,1,0], "o": [1,0,1,0,1,0],
    "p": [1,1,1,1,0,0], "q": [1,1,1,1,1,0], "r": [1,1,1,0,1,0],
    "s": [0,1,1,1,0,0], "t": [0,1,1,1,1,0], "u": [1,0,1,0,0,1],
    "v": [1,1,1,0,0,1], "w": [0,1,0,1,1,1], "x": [1,0,1,1,0,1],
    "y": [1,0,1,1,1,1], "z": [1,0,1,0,1,1],
}

# Su anki calisma kumesi: en kolay cift (en fazla nokta farkli, en az ortak)
AKTIF_HARFLER = ["g", "v"]
BRAILLE = {k: BRAILLE_TAM[k] for k in AKTIF_HARFLER}


# -------------------------------------------------------------------
def zamana_yay(stim, tekrar, cfg):
    """Uyarani pencere icinde 'tekrar' kez yineler (mors doneminden
    tasindi, gorevden bagimsiz genel bir arac — degismedi)."""
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
def braille_uyaran(harf, cfg, rng, tekrar=1, spike_sayisi=1, jitter_s=0.001, **kw):
    """
    Braille harfini, kor parmak taramasi gibi zamana yayilmis bir uyarana
    cevirir.

    6 nokta, sabit sira (1->6), esit zaman dilimi. Kabarik nokta o dilimde
    'spike_sayisi' darbe, duz nokta o dilimde TAM SESSIZLIK.

    Toplam darbe sayisi, ayni sayida kabarik noktaya sahip harfler arasinda
    (orn. g ve v, ikisi de 4 nokta) BIREBIR AYNIDIR — sayarak ayirt
    edilemez, sadece SIRA (hangi dilimde darbe var) ayirt eder.
    """
    desen = BRAILLE[harf]
    sure = cfg.stimulus_duration_s
    n_nokta = 6
    dilim = sure / n_nokta
    aktif = dilim * 0.6  # dilimin bir kismi aktif pencere, kalani dinlenme

    n_el = cfg.n_electrodes
    izler = []
    for e in range(n_el):
        nokta_idx = e % n_nokta  # n_electrodes 6'nin kati degilse de dongusel calisir
        kabarik = desen[nokta_idx]
        if not kabarik:
            izler.append(np.array([]))
            continue
        bas = nokta_idx * dilim
        if spike_sayisi <= 1:
            t = [bas + aktif * 0.3 + rng.normal(0, jitter_s)]
        else:
            t = [bas + aktif * (0.1 + 0.8 * j / (spike_sayisi - 1)) + rng.normal(0, jitter_s)
                 for j in range(spike_sayisi)]
        izler.append(np.sort(np.clip(np.array(t), 0.0, sure - 1e-4)))

    stim = StimulusPattern(spike_times=izler, n_electrodes=n_el,
                           duration_s=sure, label=harf)
    return zamana_yay(stim, tekrar, cfg)


# -------------------------------------------------------------------
def braille_etiketleri(cfg=None):
    """Su anki calisma kumesindeki harfler (varsayilan: g, v)."""
    return list(BRAILLE.keys())


def braille_config(**kw):
    """
    Braille etiketleriyle uyumlu ExperimentConfig.

    Varsayilanlar (2026-08-01, esik cesitliligi eklendikten sonra
    yeniden kalibre edildi):
        n_electrodes=12        (nokta basina 2 elektrot)
        r_membrane_mohm=20     (10 tohum, spike_sayisi=1 ile olculdu)
        v_thresh_jitter_mv=5.0 (noronlar arasi esik cesitliligi)

    ONCEKI DENEME (n_electrodes=6, homojen esik) BASARISIZDI:
    organoid ya tamamen sessizdi ya tam doygundu, arada gecerli bir
    bolge yoktu — cunku tum noronlar AYNI esige sahipti, hepsi ayni
    anda esigi asiyor ya da hicbiri asmiyordu.

    Esik cesitliligi (v_thresh_jitter_mv) eklenince gecis yumusadi ve
    R=15-25 arasinda genis, gercekten aktif (2-5 spike/trial) bir
    pencere olustu — taban dogruluk %50-56 (sans %50), std 5.5-11.4.
    R=20 bu pencerenin ortasi, en dusuk std'ye sahip nokta.
    """
    kw.setdefault("categories", braille_etiketleri())
    kw.setdefault("n_electrodes", 12)
    kw.setdefault("r_membrane_mohm", 20.0)
    kw.setdefault("v_thresh_jitter_mv", 5.0)
    return ExperimentConfig(**kw)


# Geriye donuk uyumluluk icin eski isimler (sonda/CLI kodu bunlari cagirir)
gorev_uyaran = lambda etiket, cfg, rng, indeks=0, tekrar=1, **kw: braille_uyaran(
    etiket, cfg, rng, tekrar=tekrar, **kw)
gorev_etiketleri = braille_etiketleri
gorev_config = braille_config
