"""Kaggle kernel (CPU): verify RML2018.01a against the frozen split and measure HDF5 I/O.

Clones Project RML at a pinned commit, runs scripts/prepare_rml2018.py on the real
HDF5 file, records the actual row layout and +30 dB distribution, and times the
read patterns the training loader can use. Never creates or modifies a split.
Writes /kaggle/working/rml2018_verification.json.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/murugank999900-collab/project-rml.git"
COMMIT = os.environ.get("RML_COMMIT", "f97d3bf2c39078e092b243687809f58d5eb333d2")
WORK = "/kaggle/working/project-rml"
EXPECTED_HDF5 = "/kaggle/input/datasets/tiodomm/project-rml-2018-01a/GOLD_XYZ_OSC.0001_1024.hdf5"


def sh(cmd, **kw):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=True, text=True, **kw)


def find_hdf5():
    if os.path.isfile(EXPECTED_HDF5):
        return EXPECTED_HDF5
    hits = glob.glob("/kaggle/input/**/GOLD_XYZ_OSC.0001_1024.hdf5", recursive=True)
    if not hits:
        raise SystemExit("GOLD_XYZ_OSC.0001_1024.hdf5 not found under /kaggle/input")
    return hits[0]


def meminfo_gb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) / 1024**2
    return {"MemTotal": round(info["MemTotal"], 1), "MemAvailable": round(info["MemAvailable"], 1)}


def main():
    report = {"commit_expected": COMMIT}
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    sh(f"git clone --quiet {REPO} {WORK}")
    sh(f"git -C {WORK} checkout --quiet {COMMIT}")
    head = subprocess.run(["git", "-C", WORK, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    report["commit_checked_out"] = head
    if head != COMMIT:
        raise SystemExit(f"Expected commit {COMMIT}, got {head}")

    path = find_hdf5()
    report["hdf5_path"] = path
    report["hdf5_bytes"] = os.path.getsize(path)

    # 1. The repository's own verification (split hashes, structure, per-group and +30 dB counts).
    t = time.time()
    out = subprocess.run(
        [sys.executable, "scripts/prepare_rml2018.py", "--data", path], cwd=WORK, capture_output=True, text=True
    )
    print(out.stdout, out.stderr, flush=True)
    report["prepare_rml2018"] = {
        "returncode": out.returncode,
        "stdout": out.stdout,
        "stderr": out.stderr[-4000:],
        "seconds": round(time.time() - t, 1),
    }

    # 2. Raw layout facts and I/O measurements.
    import h5py
    import numpy as np

    sys.path.insert(0, os.path.join(WORK, "src"))
    from rml.data.rml2018 import load_rml2018, load_rml2018_split

    with h5py.File(path, "r") as f:
        report["hdf5"] = {
            k: {
                "shape": list(f[k].shape),
                "dtype": str(f[k].dtype),
                "chunks": list(f[k].chunks) if f[k].chunks else None,
                "compression": f[k].compression,
            }
            for k in ("X", "Y", "Z")
        }

    ds = load_rml2018(path)
    blocks_y = ds.y.reshape(-1, 4096)[:, 0]
    blocks_snr = ds.snr.reshape(-1, 4096)[:, 0]
    snrs = sorted(set(blocks_snr.tolist()))
    class_major = bool(
        np.array_equal(blocks_y, np.repeat(np.arange(24), 26)) and np.array_equal(blocks_snr, np.tile(snrs, 24))
    )
    report["row_layout"] = {
        "class_major_snr_ascending": class_major,
        "first_blocks": [[int(a), int(b)] for a, b in zip(blocks_y[:30], blocks_snr[:30])],
    }

    split = load_rml2018_split(os.path.join(WORK, "splits/rml2018.01a_seed42.json"))
    top = {}
    for part in ("val", "test"):
        idx = getattr(split, part)
        m = ds.snr[idx] == 30
        top[part] = {"n": int(m.sum()), "per_class": np.bincount(ds.y[idx][m], minlength=24).tolist()}
    report["plus30_distribution"] = top

    io = {}
    with h5py.File(path, "r") as f:
        X = f["X"]
        for start in (0, 1_000_000, 2_400_000):
            t = time.time()
            a = X[start : start + 65536]
            dt = time.time() - t
            io[f"seq_64k_rows_at_{start}_MBps"] = round(a.nbytes / dt / 1e6, 1)
        rng = np.random.default_rng(0)
        for bs in (256, 1024):
            idx = np.sort(rng.choice(split.train, bs, replace=False))
            t = time.time()
            a = X[idx]
            io[f"random_sorted_batch_{bs}_seconds"] = round(time.time() - t, 3)
        t = time.time()
        a = X[0:262144]
        dt = time.time() - t
        io["seq_2GB_MBps"] = round(a.nbytes / dt / 1e6, 1)
    report["io"] = io
    report["machine"] = {
        "cpu_count": os.cpu_count(),
        "memory_gb": meminfo_gb(),
        "disk_free_gb": {p: round(shutil.disk_usage(p).free / 1e9, 1) for p in ("/kaggle/working", "/tmp")},
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "h5py": h5py.__version__,
    }

    report["verified"] = out.returncode == 0
    with open("/kaggle/working/rml2018_verification.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "prepare_rml2018"}, indent=2))
    if out.returncode != 0:
        raise SystemExit("prepare_rml2018.py FAILED")


if __name__ == "__main__":
    main()
