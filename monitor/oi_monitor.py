"""
oi_monitor.py — Canli deney paneli sunucusu.

Deney kodunun icinden cagrilir, her trial'da paneli besler. Panel tarayicida
acilir ve deney ilerledikce kendiliginden guncellenir.

Bagimlilik yok — sadece Python standart kutuphanesi.
Depodaki hicbir dosyayi degistirmez.

KULLANIM
--------
    from oi_monitor import Monitor

    mon = Monitor()                      # tarayiciyi acar
    mon.session(culture="fs369", note="ilk canli oturum")

    for i in range(n_trials):
        ...
        mon.trial(
            arm="A",                     # "A" = plastisite acik, "B" = kontrol
            trial=i,
            category="banana",
            predicted="apple",
            correct=False,
            electrodes=[0,1,0,1,...],    # 32 elemanli 0/1 listesi
            spikes=41,
            separability=11.4,
        )

    mon.finish()

Panel iki kolu ayri ayri biriktirir. Simulasyonda iki kolu paralel
calistirabilirsin; canli dokuda ayni dokuda donusumlu blok olarak
calistirirsin (ABAB) — panel ikisini de ayni sekilde gosterir.
"""

import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BURASI = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BURASI, "oi_panel.html")


class _Durum:
    """Panelin okudugu paylasilan durum."""

    def __init__(self):
        self.kilit = threading.Lock()
        self.sifirla()

    def sifirla(self):
        with self.kilit:
            self.veri = {
                "live": True,
                "running": False,
                "finished": False,
                "culture": "—",
                "note": "",
                "started": None,
                "trial": 0,
                "stimulus": None,
                "arms": {
                    "A": {"label": "plasticity on", "hist": [], "win": [],
                          "acc": 0.0, "sep": None, "electrodes": [0] * 32,
                          "spikes": 0, "predicted": None, "correct": None},
                    "B": {"label": "control", "hist": [], "win": [],
                          "acc": 0.0, "sep": None, "electrodes": [0] * 32,
                          "spikes": 0, "predicted": None, "correct": None},
                },
            }

    def anlik(self):
        with self.kilit:
            return json.dumps(self.veri)


DURUM = _Durum()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # sunucu gurultusunu sustur

    def _gonder(self, kod, tip, govde):
        self.send_response(kod)
        self.send_header("Content-Type", tip)
        self.send_header("Content-Length", str(len(govde)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(govde)

    def do_GET(self):
        yol = self.path.split("?")[0]
        if yol == "/api/state":
            self._gonder(200, "application/json", DURUM.anlik().encode("utf-8"))
        elif yol in ("/", "/index.html"):
            if not os.path.exists(PANEL):
                self._gonder(500, "text/plain; charset=utf-8",
                             ("oi_panel.html bulunamadi. Bu dosyayla ayni "
                              "klasorde olmali:\n" + PANEL).encode("utf-8"))
                return
            with open(PANEL, "rb") as f:
                self._gonder(200, "text/html; charset=utf-8", f.read())
        else:
            self._gonder(404, "text/plain", b"yok")


class Monitor:
    """Canli panel. Deney kodundan cagrilir."""

    def __init__(self, port=8765, open_browser=True, quiet=False):
        self.port = port
        self.quiet = quiet
        DURUM.sifirla()
        self._sunucu = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        self._t = threading.Thread(target=self._sunucu.serve_forever, daemon=True)
        self._t.start()
        self.url = "http://127.0.0.1:%d/" % port
        if not quiet:
            print("[monitor] panel hazir: %s" % self.url)
        if open_browser:
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        time.sleep(0.4)  # tarayicinin baglanmasina firsat ver

    # ---------------------------------------------------------------
    def session(self, culture="—", note=""):
        with DURUM.kilit:
            DURUM.veri["culture"] = culture
            DURUM.veri["note"] = note
            DURUM.veri["started"] = time.time()
            DURUM.veri["running"] = True
            DURUM.veri["finished"] = False

    def stimulus(self, category, pixels=None):
        """Sunulan uyarani panele bildir. pixels: 64 elemanli 0-1 listesi."""
        with DURUM.kilit:
            DURUM.veri["stimulus"] = {
                "category": category,
                "pixels": list(pixels) if pixels is not None else None,
            }

    def trial(self, arm, trial, category, predicted, correct,
              electrodes=None, spikes=0, separability=None, window=30):
        """Bir trial sonucunu panele gonder."""
        arm = str(arm).upper()
        if arm not in ("A", "B"):
            raise ValueError("arm 'A' veya 'B' olmali")
        with DURUM.kilit:
            k = DURUM.veri["arms"][arm]
            k["win"].append(1 if correct else 0)
            if len(k["win"]) > window:
                k["win"] = k["win"][-window:]
            k["acc"] = 100.0 * sum(k["win"]) / len(k["win"])
            k["hist"].append(round(k["acc"], 2))
            k["predicted"] = predicted
            k["correct"] = bool(correct)
            k["spikes"] = int(spikes)
            if electrodes is not None:
                k["electrodes"] = [1 if x else 0 for x in electrodes]
            if separability is not None:
                k["sep"] = float(separability)
            DURUM.veri["trial"] = max(DURUM.veri["trial"], int(trial) + 1)
            DURUM.veri["stimulus"] = DURUM.veri.get("stimulus") or {}
            if isinstance(DURUM.veri["stimulus"], dict):
                DURUM.veri["stimulus"].setdefault("category", category)

    def separability(self, arm, value):
        with DURUM.kilit:
            DURUM.veri["arms"][str(arm).upper()]["sep"] = float(value)

    def finish(self, keep_open=True):
        with DURUM.kilit:
            DURUM.veri["running"] = False
            DURUM.veri["finished"] = True
        if not self.quiet:
            print("[monitor] oturum bitti. panel acik: %s" % self.url)
        if keep_open:
            try:
                input("[monitor] paneli kapatmak icin Enter'a bas...")
            except (EOFError, KeyboardInterrupt):
                pass
        self.close()

    def close(self):
        try:
            self._sunucu.shutdown()
        except Exception:
            pass


# -------------------------------------------------------------------
if __name__ == "__main__":
    # Sunucuyu tek basina denemek icin
    m = Monitor()
    m.session(culture="test", note="oi_monitor kendi kendine test")
    import math
    import random
    for i in range(120):
        for arm in ("A", "B"):
            ok = random.random() < 0.33 + 0.34 * (1 - math.exp(-i / 25.0))
            m.trial(arm=arm, trial=i, category=["banana", "apple", "pear"][i % 3],
                    predicted="apple", correct=ok,
                    electrodes=[1 if random.random() < 0.4 else 0 for _ in range(32)],
                    spikes=random.randint(10, 60),
                    separability=11.6 - 1.5 * (1 - math.exp(-i / 50.0)))
        time.sleep(0.15)
    m.finish()
