"""
oi_signal.py — organoid_oi_v2
==============================
Image → spike pattern conversion.

v2 changes (peer review fix #3):
    - Difference of Gaussians (DoG) filter replaces raw luminance
    - Rank-order (latency) coding replaces rate coding
    - Separate on/off channels for contrast polarity

DoG filter (center-surround):
    DoG(x,y) = G(σ1) - G(σ2)
    Positive values → on-center response (bright features)
    Negative values → off-center response (dark features on bright bg)

Rank-order coding (Van Rullen & Thorpe 2001):
    Most salient pixel fires first (latency = 0 ms)
    Least salient pixel fires last (latency = max_latency_ms)
    Latency = max_latency * (1 - normalized_salience)

    STDP is highly sensitive to spike order — early spikes have
    exponentially more influence than late ones. This means the most
    prominent visual features drive learning most strongly.

References:
    Van Rullen & Thorpe (2001). Rate coding vs temporal order coding.
    Neural Computation, 13(6), 1255-1283.

    Masquelier & Thorpe (2007). Unsupervised learning of visual features
    through STDP. PLoS Computational Biology, 3(2), e31.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import sys

sys.path.insert(0, str(Path(__file__).parent))
from oi_types import StimulusPattern, ExperimentConfig


# ─────────────────────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────────────────────

def load_image(path: str, size: int = 8) -> np.ndarray:
    """Load image → grayscale luminance [size×size], values in [0,1]."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required: pip install Pillow")
    img = Image.open(path).convert('L')
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=float) / 255.0


def luminance_from_array(arr: np.ndarray, size: int = 8) -> np.ndarray:
    """Convert numpy array (RGB or gray) to resized grayscale luminance."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL required: pip install Pillow")
    if arr.ndim == 3:
        lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    else:
        lum = arr.astype(float)
    lum = lum / lum.max() if lum.max() > 0 else lum
    img = Image.fromarray((lum * 255).astype(np.uint8))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=float) / 255.0


# ─────────────────────────────────────────────────────────────
# DIFFERENCE OF GAUSSIANS (DoG) FILTER
# ─────────────────────────────────────────────────────────────

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """2D Gaussian kernel, normalized to sum=1."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def dog_filter(luminance: np.ndarray,
               sigma1: float = 1.0,
               sigma2: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Difference of Gaussians filter to luminance map.

    Returns:
        on_response  : positive DoG values (bright features)
        off_response : negative DoG values inverted (dark features)

    Both are clipped to [0, 1].
    """
    from scipy.ndimage import convolve
    size = luminance.shape[0]

    k1 = gaussian_kernel(size, sigma1)
    k2 = gaussian_kernel(size, sigma2)

    dog = convolve(luminance, k1) - convolve(luminance, k2)

    # Separate on/off channels
    on = np.clip(dog, 0, None)
    off = np.clip(-dog, 0, None)

    # Normalize each to [0, 1]
    on = on / on.max() if on.max() > 0 else on
    off = off / off.max() if off.max() > 0 else off

    return on, off


# ─────────────────────────────────────────────────────────────
# RANK-ORDER (LATENCY) CODING
# ─────────────────────────────────────────────────────────────

def rank_order_encode(
    salience: np.ndarray,
    duration_s: float,
    max_latency_ms: float = 50.0,
    n_electrodes: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Convert salience map to rank-order spike times.

    Most salient electrode fires first (latency ≈ 0).
    Least salient fires last (latency ≈ max_latency_ms).
    Zero-salience electrodes do not fire.

    Args:
        salience     : [n_electrodes] or [H×W] salience values in [0, 1]
        duration_s   : stimulus duration
        max_latency_ms: maximum spike latency
        n_electrodes : expected electrode count (validation)

    Returns:
        spike_times : list of arrays, one per electrode
    """
    flat = salience.flatten()
    n = len(flat)

    max_lat_s = max_latency_ms / 1000.0
    spike_times = []

    for sal in flat:
        if sal <= 0.01:
            spike_times.append(np.array([]))
        else:
            # Inverse: high salience → short latency
            latency = max_lat_s * (1.0 - sal)
            if latency < duration_s:
                spike_times.append(np.array([latency]))
            else:
                spike_times.append(np.array([]))

    return spike_times


# Import List for type hint


def rate_encode_sparse(
    salience: np.ndarray,
    duration_s: float,
    max_rate_hz: float = 100.0,
    rng: np.random.Generator = None,
) -> List[np.ndarray]:
    """
    Fallback: Poisson rate encoding (v1 style).
    Used when use_rank_order=False.
    """
    if rng is None:
        rng = np.random.default_rng()
    flat = salience.flatten()
    dt_s = 0.001
    n_bins = int(duration_s / dt_s)
    times = np.arange(n_bins) * dt_s
    spike_times = []
    for sal in flat:
        rate = sal * max_rate_hz
        p = min(rate * dt_s, 1.0)
        mask = rng.random(n_bins) < p
        spike_times.append(times[mask])
    return spike_times


# ─────────────────────────────────────────────────────────────
# MAIN ENCODING
# ─────────────────────────────────────────────────────────────

def encode_image(
    image_path: str,
    label: str,
    config: ExperimentConfig = None,
    rng: np.random.Generator = None,
) -> StimulusPattern:
    """
    Full pipeline: image file → StimulusPattern (v2).

    Steps:
        1. Load and resize image
        2. Apply DoG filter → on + off channels
        3. Concatenate: first n_electrodes/2 = on, rest = off
           (if n_electrodes < 2*pixels, use on channel only)
        4. Rank-order or rate encoding
    """
    if config is None:
        config = ExperimentConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    luminance = load_image(image_path, size=config.image_size)
    return _encode_luminance(luminance, label, config, rng, source=image_path)


def encode_array(
    arr: np.ndarray,
    label: str,
    config: ExperimentConfig = None,
    rng: np.random.Generator = None,
) -> StimulusPattern:
    """Encode numpy array → StimulusPattern (v2)."""
    if config is None:
        config = ExperimentConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)
    luminance = luminance_from_array(arr, size=config.image_size)
    return _encode_luminance(luminance, label, config, rng, source='array')


def _encode_luminance(
    luminance: np.ndarray,
    label: str,
    config: ExperimentConfig,
    rng: np.random.Generator,
    source: str = '',
) -> StimulusPattern:
    """Internal: luminance map → StimulusPattern."""
    on, off = dog_filter(luminance, config.dog_sigma1, config.dog_sigma2)

    n_pixels = config.image_size ** 2
    n_elec = config.n_electrodes

    # Use on-channel for first half, off-channel for second half
    if n_elec >= 2 * n_pixels:
        salience = np.concatenate([on.flatten(), off.flatten()])
        salience = salience[:n_elec]
    elif n_elec >= n_pixels:
        n_off = n_elec - n_pixels
        salience = np.concatenate([on.flatten(), off.flatten()[:n_off]])
    else:
        salience = on.flatten()[:n_elec]

    if config.use_rank_order:
        spike_times = rank_order_encode(
            salience, config.stimulus_duration_s, config.max_latency_ms
        )
    else:
        spike_times = rate_encode_sparse(
            salience, config.stimulus_duration_s, config.max_spike_rate_hz, rng
        )

    return StimulusPattern(
        spike_times=spike_times,
        n_electrodes=n_elec,
        duration_s=config.stimulus_duration_s,
        label=label,
        meta={
            'source': source,
            'image_size': config.image_size,
            'encoding': 'rank_order' if config.use_rank_order else 'rate',
            'mean_luminance': float(luminance.mean()),
            'on_active': int((on > 0.01).sum()),
            'off_active': int((off > 0.01).sum()),
            'total_spikes': sum(len(s) for s in spike_times),
        }
    )


def synthetic_stimulus(
    label: str,
    pattern: str = 'random',
    config: ExperimentConfig = None,
    rng: np.random.Generator = None,
) -> StimulusPattern:
    """
    Synthetic stimulus without image file. For testing.

    Patterns: 'random', 'bright', 'dark', 'center', 'stripe', 'edge'
    """
    if config is None:
        config = ExperimentConfig()
    if rng is None:
        rng = np.random.default_rng(config.random_seed)

    s = config.image_size

    if pattern == 'random':
        arr = rng.random((s, s))
    elif pattern == 'bright':
        arr = np.ones((s, s))
    elif pattern == 'dark':
        arr = np.zeros((s, s))
    elif pattern == 'center':
        arr = np.zeros((s, s))
        cx, cy = s // 2, s // 2
        for r in range(s):
            for c in range(s):
                dist = np.sqrt((r-cx)**2 + (c-cy)**2)
                arr[r, c] = np.exp(-dist**2 / (2*(s/4)**2))
    elif pattern == 'stripe':
        arr = np.zeros((s, s))
        arr[:, ::2] = 1.0
    elif pattern == 'edge':
        arr = np.zeros((s, s))
        arr[:, :s//2] = 1.0      # left half bright, right half dark
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    return _encode_luminance(arr, label, config, rng, source=f'synthetic:{pattern}')

