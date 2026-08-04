"""
hardware/finalspark_organoid.py — Hardware adapter for FinalSpark's
Neuroplatform, implementing the SAME interface as sim.oi_synth.SyntheticOrganoid.

WHY THIS FILE EXISTS
---------------------
Every simulation-side script (sentence_demo.py, calibrate.py, and the whole
organoid-oi framework) was built against one contract:

    organoid.respond(stimulus, timestamp, post_stimulus_current=None) -> response

`response` must expose `.spike_times` — a list of arrays, one per neuron/unit,
each array the spike timestamps (in seconds, relative to the trial start)
that unit produced.

If this file correctly implements that same contract against FinalSpark's
real API, every script above runs UNCHANGED on real tissue — only the
`--backend hardware` flag changes.

STATUS — READ THIS BEFORE USING
--------------------------------
This is a STRUCTURAL adapter, not a tested one. The three methods below
(`_stimulate`, `_read_spikes`, `_inject_current`) contain the exact shape of
call FinalSpark's Neuroplatform API is documented to expose, based on the
platform's public description — but they have NOT been run against a live
session, because this project has not yet been granted (or has not yet used)
live API credentials. Each is marked with a TODO at the exact line where the
real API call goes. Treat every number in this file (timing conversions,
channel indexing) as a first draft to verify against FinalSpark's actual
documentation before running a paid session.

WHAT TO DO BEFORE THE FIRST REAL SESSION
------------------------------------------
1. Confirm the exact method names / endpoint shapes in FinalSpark's current
   SDK or REST API documentation (this may have changed since this file was
   written) and fill in the three TODOs.
2. Run calibrate.py --backend hardware --electrodes-only FIRST (see that
   script) — a short, cheap, non-destructive check that electrode addressing
   is correct, before spending paid session time on the full demo.
3. Confirm the platform's channel count for your session (this project
   assumes up to 32, matching FinalSpark's commonly cited configuration —
   confirm this for your specific culture/session before running).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class _HardwareResponse:
    """Matches sim.oi_synth's OrganoidResponse shape exactly."""
    spike_times: List[np.ndarray]
    n_neurons: int
    duration_s: float
    timestamp: float


class FinalSparkOrganoid:
    """
    Drop-in replacement for sim.oi_synth.SyntheticOrganoid.

    Construct with the same ExperimentConfig used elsewhere in this project,
    plus FinalSpark session credentials. `respond()` has the identical
    signature and return shape as SyntheticOrganoid.respond().
    """

    def __init__(self, config, session=None, api_key: Optional[str] = None,
                culture_id: Optional[str] = None, rng=None):
        self.config = config
        self.n_neurons = getattr(config, "n_neurons", None)  # units recorded, not simulated
        self.rng = rng or np.random.default_rng(0)
        self.api_key = api_key
        self.culture_id = culture_id
        self._session = session
        if session is None and (api_key is None or culture_id is None):
            raise ValueError(
                "FinalSparkOrganoid needs either a live `session` object, or "
                "both `api_key` and `culture_id` to open one. Nothing is "
                "connected yet at construction time — see module docstring."
            )
        # TODO: if session is None, open one here using FinalSpark's client,
        # e.g.:  self._session = finalspark.connect(api_key=api_key, culture=culture_id)
        # Left unimplemented intentionally — fill in against current SDK docs.

    # ------------------------------------------------------------------
    def respond(self, stimulus, timestamp: float,
               post_stimulus_current: Optional[np.ndarray] = None,
               dt_ms: float = 1.0) -> _HardwareResponse:
        """
        Same two-phase contract as SyntheticOrganoid.respond():
          Phase 1 — deliver `stimulus` (per-electrode spike times), record
                    the culture's response for stimulus.duration_s.
          Phase 2 — if `post_stimulus_current` is given (reward/penalty
                    waveform), deliver it, record the response for its
                    duration too. Concatenate and return both phases,
                    exactly like the synthetic organoid does — so
                    downstream STDP/decoder code needs no changes.
        """
        self._stimulate(stimulus)
        phase1 = self._read_spikes(duration_s=stimulus.duration_s)

        phase2 = []
        total_duration = stimulus.duration_s
        if post_stimulus_current is not None and len(post_stimulus_current) > 0:
            self._inject_current(post_stimulus_current, dt_ms=dt_ms)
            extra_duration = len(post_stimulus_current) * dt_ms / 1000.0
            phase2 = self._read_spikes(duration_s=extra_duration)
            phase2 = [t + stimulus.duration_s for t in phase2]  # offset into combined timeline
            total_duration += extra_duration

        combined = [np.concatenate([p1, p2]) if len(phase2) else p1
                   for p1, p2 in zip(phase1, phase2 or [np.array([])] * len(phase1))]

        return _HardwareResponse(
            spike_times=combined,
            n_neurons=len(combined),
            duration_s=total_duration,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    def _stimulate(self, stimulus) -> None:
        """
        Deliver `stimulus.spike_times` (one array per electrode, in seconds
        relative to trial start) to the physical MEA electrodes.

        TODO: replace with FinalSpark's real stimulation call. Expected
        shape, based on the platform's documented closed-loop design:

            for electrode_idx, spike_times in enumerate(stimulus.spike_times):
                if len(spike_times) == 0:
                    continue
                self._session.stimulate(
                    channel=electrode_idx,
                    timestamps_ms=[t * 1000 for t in spike_times],
                    amplitude_uA=...,   # confirm safe amplitude range with FinalSpark docs
                )
        """
        raise NotImplementedError(
            "FinalSparkOrganoid._stimulate is a structural placeholder. "
            "Fill in the real API call before running against live tissue."
        )

    def _read_spikes(self, duration_s: float) -> List[np.ndarray]:
        """
        Read back recorded spike times for the last `duration_s` seconds,
        one array per channel, in seconds relative to the read window start.

        TODO: replace with FinalSpark's real readback call. Expected shape:

            raw = self._session.read_activity(duration_ms=duration_s * 1000)
            return [np.array(raw[ch]) / 1000.0 for ch in range(n_channels)]
        """
        raise NotImplementedError(
            "FinalSparkOrganoid._read_spikes is a structural placeholder. "
            "Fill in the real API call before running against live tissue."
        )

    def _inject_current(self, waveform: np.ndarray, dt_ms: float) -> None:
        """
        Deliver a reward/penalty current waveform (the same waveform
        core/oi_reward.py already generates for the synthetic organoid —
        10 Hz sub-threshold theta for reward, 200 Hz noise for penalty).

        TODO: replace with FinalSpark's real current-injection call.
        """
        raise NotImplementedError(
            "FinalSparkOrganoid._inject_current is a structural placeholder. "
            "Fill in the real API call before running against live tissue."
        )
