"""
sonda_recurrent.py — Organoid ici tekrarlayan baglanti hipotezi.

HIPOTEZ: Mevcut model tamamen ileri beslemeli (elektrot -> noron). Noronlar
birbirini hic gormuyor. Gercek organoidde ag kendi ic dinamigine sahip ve
STDP'nin sekillendirdigi esas yapi o. Ileri beslemeli tek katmanda decoder
her seyi telafi ettigi icin STDP'ye is kalmiyor olabilir.

TEST: Noronlar arasi seyrek, E/I dengeli tekrarlayan baglanti ekle.
STDP acik/kapali dogruluk farki olusuyor mu?

Depodaki hicbir dosyayi degistirmez — SyntheticOrganoid alt sinifidir.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.resolve()
for p in ["", "core", "sim", "analysis"]:
    sys.path.insert(0, str(ROOT / p))

import numpy as np
from core.oi_types import ExperimentConfig, OrganoidResponse
from core.oi_signal import synthetic_stimulus
from core.oi_loop import ClosedLoopExperiment
from sim.oi_synth import SyntheticOrganoid, psp_kernel

DESEN = ["center", "edge", "stripe"]


class RecurrentOrganoid(SyntheticOrganoid):
    """
    Ileri besleme + noronlar arasi tekrarlayan baglanti.

    p_rec     : noronlar arasi baglanma olasiligi
    g_rec     : tekrarlayan baglanti kazanci (0 = mevcut ileri beslemeli model)
    frac_inh  : inhibitor noron orani (kortekste ~%20)
    """
    def __init__(self, config=None, rng=None, p_rec=0.10, g_rec=0.0, frac_inh=0.2):
        super().__init__(config, rng=rng)
        n = self.n_neurons
        self.g_rec = g_rec
        self.inh = self.rng.random(n) < frac_inh
        maske = self.rng.random((n, n)) < p_rec
        np.fill_diagonal(maske, False)
        W = self.rng.random((n, n))
        W[self.inh, :] *= -4.0            # inhibitor satirlar negatif ve daha guclu
        self.W_rec = np.where(maske, W, 0.0) * g_rec

    def _surus_hepsi(self, stimulus, t):
        """Tum noronlar icin elektrot surusu — [n_t, n_neurons]."""
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

    def _simule(self, I_dis, dt_s):
        """Tekrarlayan baglantili vektorize LIF. Donen: spike zamanlari listesi."""
        c = self.config
        n_t, n = I_dis.shape
        v = np.full(n, c.v_rest_mv, dtype=float)
        refrak = np.zeros(n, dtype=int)
        refrak_adim = max(1, int(c.refractory_ms / 1000.0 / dt_s))
        tau_s = c.tau_membrane_ms / 1000.0
        tau_syn_s = c.tau_syn_ms / 1000.0
        sonum = np.exp(-dt_s / tau_syn_s)
        I_rec = np.zeros(n)
        atesler = [[] for _ in range(n)]

        for k in range(n_t):
            I = I_dis[k] + I_rec
            aktif = refrak == 0
            dv = (-(v - c.v_rest_mv) + c.r_membrane_mohm * I) * (dt_s / tau_s)
            v = np.where(aktif, v + dv, c.v_reset_mv)
            refrak = np.where(aktif, refrak, refrak - 1)
            atesleyen = aktif & (v >= c.v_thresh_mv)
            if atesleyen.any():
                v[atesleyen] = c.v_reset_mv
                refrak[atesleyen] = refrak_adim
                for j in np.where(atesleyen)[0]:
                    atesler[j].append(k * dt_s)
            # tekrarlayan akim: 1 adim gecikme + sinaptik sonum
            I_rec = I_rec * sonum + self.W_rec.T @ atesleyen.astype(float)

        return [np.array(a) for a in atesler]

    def respond(self, stimulus, timestamp, post_stimulus_current=None, dt_ms=1.0):
        dt_s = dt_ms / 1000.0
        n1 = int(stimulus.duration_s / dt_s)
        t1 = np.arange(n1) * dt_s
        I1 = self._surus_hepsi(stimulus, t1)
        I1 += self.rng.normal(0, self._noise_current_na, I1.shape)
        z1 = self._simule(I1, dt_s)

        z2 = [np.array([]) for _ in range(self.n_neurons)]
        toplam = stimulus.duration_s
        if post_stimulus_current is not None and len(post_stimulus_current) > 0:
            n2 = len(post_stimulus_current)
            I2 = np.tile(post_stimulus_current[:, None], (1, self.n_neurons))
            I2 = I2 + self.rng.normal(0, self._noise_current_na, I2.shape)
            z2 = [a + stimulus.duration_s for a in self._simule(I2, dt_s)]
            toplam += n2 * dt_s

        return OrganoidResponse(
            spike_times=[np.concatenate([a, b]) for a, b in zip(z1, z2)],
            n_neurons=self.n_neurons,
            duration_s=toplam,
            timestamp=timestamp,
        )


def ayrilabilirlik(org, cfg, tohum=999, n=5):
    rng = np.random.default_rng(tohum)
    kat = {k: [] for k in cfg.categories}
    for i in range(n * 3):
        k = cfg.categories[i % 3]
        r = org.respond(synthetic_stimulus(k, DESEN[i % 3], cfg, rng), timestamp=0.0)
        kat[k].append(np.array([len(x) for x in r.spike_times], float))
    ort = {k: np.mean(v, axis=0) for k, v in kat.items()}
    ic = np.mean([np.mean([np.linalg.norm(v - ort[k]) for v in kat[k]]) for k in cfg.categories])
    ks = list(cfg.categories)
    dis = np.mean([np.linalg.norm(ort[ks[a]] - ort[ks[b]])
                   for a in range(3) for b in range(a + 1, 3)])
    return dis / (ic + 1e-9)


def kos(g_rec, lr, n_trial=100, tohum=7, n_neurons=60):
    cfg = ExperimentConfig(learning_rate=lr, n_neurons=n_neurons)
    rng = np.random.default_rng(tohum)
    org = RecurrentOrganoid(cfg, rng=rng, g_rec=g_rec)
    once = ayrilabilirlik(org, cfg)
    ex = ClosedLoopExperiment(org, cfg, rng=rng)
    for i in range(n_trial):
        ex.run_trial(synthetic_stimulus(cfg.categories[i % 3], DESEN[i % 3], cfg, rng), i)
    sonra = ayrilabilirlik(org, cfg)
    acc = 100 * np.mean([t.correct for t in ex.trial_history[-40:]])
    sp = np.mean([len(np.concatenate(t.response.spike_times))
                  if getattr(t, "response", None) is not None else 0
                  for t in ex.trial_history[-10:]])
    return acc, once, sonra


print("=" * 74)
print("  TEKRARLAYAN BAGLANTI SONDASI")
print("=" * 74)
print("  g_rec = 0.0  ->  mevcut ileri beslemeli model (kontrol)")
print("  60 noron, 100 trial, son 40 trial dogrulugu, sans %33.3")
print()
print(f"  {'g_rec':>7}{'STDP kapali':>14}{'STDP acik':>12}{'fark':>9}"
      f"{'ayrilab. STDP+':>16}{'ayrilab. STDP-':>16}")
print("  " + "-" * 72)

for g in [0.0, 0.5, 1.0, 2.0]:
    a, o0, s0 = kos(g, 0.0)
    b, o1, s1 = kos(g, 0.01)
    fark = b - a
    net = (s1 - o1) - (s0 - o0)
    isaret = "  <-- ETKILI" if abs(fark) > 2.0 else ""
    print(f"  {g:>7.1f}{a:>13.1f}%{b:>11.1f}%{fark:>+9.1f}"
          f"{s1 - o1:>+16.3f}{s0 - o0:>+16.3f}{isaret}")

print()
print("  ayrilab. = egitim sonrasi - oncesi (decoder-bagimsiz)")
print("  Karar: fark > 2 puan VE STDP+ ayrilabilirlik STDP-'den yuksekse hipotez dogrulanir.")
