"""Kaggle kernel (GPU T4): run one RML2018.01a experiment from a pinned commit.

Settings (edit below or pass as environment variables):
  RML_COMMIT   commit to check out (must be pushed to GitHub)
  RML_CONFIG   experiment config, e.g. configs/rml2018/exp001_cldnn.yaml
  RML_MODE     "smoke" (1 epoch x 200 batches, throughput/memory check, no registry row) or "train"
  RML_RESUME   optional: run directory to continue, e.g. copied from a previous kernel's output
               (/kaggle/input/<kernel>/project-rml/experiments/rml2018.01a/<run_id>)

Everything is written under /kaggle/working/project-rml (saved as kernel output):
the run directory (config, env, history, per-epoch val metrics, checkpoints)
and results/registry.csv. Copy the run directory (without checkpoints) and the
new registry row back into the repository to archive the experiment.
The test split is not evaluated here.
"""

import os
import shutil
import subprocess
import sys

REPO = "https://github.com/murugank999900-collab/project-rml.git"
COMMIT = os.environ.get("RML_COMMIT", "REPLACE_WITH_PUSHED_COMMIT")
CONFIG = os.environ.get("RML_CONFIG", "configs/rml2018/exp001_cldnn.yaml")
MODE = os.environ.get("RML_MODE", "smoke")
RESUME = os.environ.get("RML_RESUME", "")
DATA = "/kaggle/input/datasets/tiodomm/project-rml-2018-01a/GOLD_XYZ_OSC.0001_1024.hdf5"
WORK = "/kaggle/working/project-rml"


def sh(cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def main():
    data = DATA
    if not os.path.isfile(data):
        import glob

        hits = glob.glob("/kaggle/input/**/GOLD_XYZ_OSC.0001_1024.hdf5", recursive=True)
        if not hits:
            raise SystemExit("GOLD_XYZ_OSC.0001_1024.hdf5 not found under /kaggle/input")
        data = hits[0]

    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    sh(["git", "clone", "--quiet", REPO, WORK])
    sh(["git", "checkout", "--quiet", COMMIT], cwd=WORK)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORK, capture_output=True, text=True).stdout.strip()
    if not head.startswith(COMMIT):
        raise SystemExit(f"Checked out {head}, expected {COMMIT}")
    sh(["nvidia-smi"])

    cmd = [sys.executable, "scripts/train_rml2018.py", "--data", data]
    if RESUME:
        target = os.path.join(WORK, "experiments", "rml2018.01a", os.path.basename(RESUME.rstrip("/")))
        shutil.copytree(RESUME, target)
        cmd += ["--resume", target]
    else:
        cmd += ["--config", CONFIG] + (["--smoke"] if MODE == "smoke" else [])
    sh(cmd, cwd=WORK)


if __name__ == "__main__":
    main()
