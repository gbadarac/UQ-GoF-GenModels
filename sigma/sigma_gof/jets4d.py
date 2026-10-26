"""
4D jet-embedding benchmark: a finite pool of ~2M QCD-jet events in a 4D
contrastive embedding, partitioned once into a test region and a training
bank.

Partition (seed MASTER = 777): one fixed permutation of the pool; the first
N_TEST_REGION = 1M entries form the test region, the remaining ~1M the
training bank.
  training sample of fit b: n events drawn without replacement from the
      training bank with RandomState(SEED_MODEL_BASE + b). Different fits
      therefore overlap by ~n^2/1M events.
  observed set s at size nte for fit b: nte events drawn without replacement
      from the whole test region, seeded by (b, s, nte). Draws are distinct
      across fits, sets and N_test values, but OVERLAP (finite pool):
      statistics pooled over sets carry the finite-pool floor
      sd(mean z) = sqrt(rho + (1 - rho)/n), rho = nte / 1M.

Data location: every *.npz in the directory SIGMA_JETS_DATA (default
<project root>/data/jets4d/), each holding an `embeddings` array of shape
(n, 4), concatenated in sorted file-name order (the pool permutation, and
hence every training and observed set, depends on that order).
"""

import glob
import os

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)            # the snapshot root
DATA_DIR = os.environ.get("SIGMA_JETS_DATA",
                          os.path.join(PROJECT_ROOT, "data", "jets4d"))

MASTER = 777
N_TEST_SETS = 40
N_TEST_BASE = 25_000
N_TEST_REGION = N_TEST_SETS * N_TEST_BASE      # 1M-event test region
SEED_OBS_BASE = 606_000
SEED_MODEL_BASE = 900_000
N_TRAIN = 100_000

_cache = {}


def load_pool():
    """All embedding files concatenated into one (N_pool, 4) float64 array
    (cached)."""
    if "pool" not in _cache:
        files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
        if not files:
            raise FileNotFoundError(
                f"No npz files under {DATA_DIR} — the jets4d benchmark needs "
                f"the embedding files (set SIGMA_JETS_DATA).")
        _cache["pool"] = np.concatenate(
            [np.load(f)["embeddings"] for f in files]).astype(np.float64)
    return _cache["pool"]


def _carve():
    if "tests" not in _cache:
        pool = load_pool()
        perm = np.random.RandomState(MASTER).permutation(pool.shape[0])
        _cache["tests"] = perm[:N_TEST_REGION].reshape(N_TEST_SETS, N_TEST_BASE)
        _cache["bank"] = perm[N_TEST_REGION:]
    return _cache["pool"], _cache["tests"], _cache["bank"]


def train_data(b, n=N_TRAIN):
    """Training sample of fit b (n events from the training bank)."""
    pool, _, bank = _carve()
    rs = np.random.RandomState(SEED_MODEL_BASE + int(b))
    return pool[rs.choice(bank, size=int(n), replace=False)]


def observed_data(s, nte, b):
    """Observed set s at size nte for fit b: an nte-subset of the whole 1M
    test region drawn without replacement, seeded by (b, s, nte)."""
    if nte > N_TEST_REGION:
        raise ValueError(f"nte={nte} exceeds the {N_TEST_REGION} test region")
    pool, tests, _ = _carve()
    region = tests.ravel()
    rng = np.random.default_rng([SEED_OBS_BASE, int(b), int(s), int(nte)])
    return pool[rng.choice(region, size=int(nte), replace=False)]


def heldout_events():
    """All events of the 1M test region (in test-region order); every
    observed set is a subset of these."""
    pool, tests, _ = _carve()
    return pool[tests.ravel()]
