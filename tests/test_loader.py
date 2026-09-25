import pickle

import numpy as np
import pytest

from conftest import MODS, SNRS, make_groups
from rml.data import from_groups, load_rml2016, validate_structure


def test_load_pickle_preserves_shape_labels_and_snr(tmp_path):
    groups = make_groups()
    # Mimic Python-2 era pickles: bytes modulation names and float64 samples.
    raw = {(m.encode(), s): a.astype(np.float64) for (m, s), a in groups.items()}
    path = tmp_path / "RML2016.10a_dict_optimized.pkl"
    path.write_bytes(pickle.dumps(raw, protocol=2))

    ds = load_rml2016(path)
    assert ds.X.shape == (len(MODS) * len(SNRS) * 20, 2, 128)
    assert ds.X.dtype == np.float32
    assert ds.classes == tuple(sorted(MODS))
    assert ds.snrs == tuple(sorted(SNRS))

    # Canonical order: sorted by (mod, snr), original within-group order.
    offset = 0
    for mod in sorted(MODS):
        for snr in sorted(SNRS):
            block = slice(offset, offset + 20)
            np.testing.assert_allclose(ds.X[block], groups[(mod, snr)], rtol=1e-6)
            assert np.all(ds.y[block] == ds.classes.index(mod))
            assert np.all(ds.snr[block] == snr)
            offset += 20


def test_rejects_wrong_sample_shape():
    with pytest.raises(ValueError, match="shape"):
        from_groups({("BPSK", 0): np.zeros((5, 128, 2), dtype=np.float32)})


def test_rejects_non_dict_pickle(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="dict"):
        load_rml2016(path)


def test_validate_structure():
    ds = from_groups(make_groups())
    validate_structure(ds, num_classes=3, snrs=SNRS, samples_per_group=20)
    with pytest.raises(ValueError, match="classes"):
        validate_structure(ds, num_classes=11)
    with pytest.raises(ValueError, match="SNRs"):
        validate_structure(ds, snrs=[0, 18])
    with pytest.raises(ValueError, match="samples"):
        validate_structure(ds, samples_per_group=1000)
