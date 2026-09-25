import numpy as np

MODS = ["BPSK", "QPSK", "AM-DSB"]
SNRS = [-10, 0, 6, 18]
N_PER_GROUP = 20


def make_groups(mods=MODS, snrs=SNRS, n=N_PER_GROUP, seed=0):
    """Synthetic {(mod, snr): (n, 2, 128) float32} dict."""
    rng = np.random.default_rng(seed)
    return {(mod, snr): rng.standard_normal((n, 2, 128)).astype(np.float32) for mod in mods for snr in snrs}
