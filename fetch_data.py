"""Fetch and verify the trace release of Lugoloobi et al. (arXiv:2605.14786).

Downloads the authors' Git LFS tarball (github.com/KabakaWilliam/known_actions)
plus two reference files, verifies the archive checksum, unpacks it, and writes
a per-agent inventory checked against the paper's Table 1 task counts. The
release carries no license, so nothing from it is committed to this repo; this
script recreates data/ locally.
"""

import ssl
import sys
import tarfile
import urllib.request

import certifi
import pandas as pd

from common import ROOT, dataset_base, is_valid_trace, iter_trace_paths, load_config, load_trace, sha256_file, split_column


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def ensure_release(cfg):
    rel = cfg["release"]
    tarball = ROOT / rel["tarball_path"]
    if not tarball.exists() or tarball.stat().st_size != rel["tarball_bytes"]:
        download(rel["tarball_url"], tarball)
    digest = sha256_file(tarball)
    if digest != rel["tarball_sha256"]:
        sys.exit(f"checksum mismatch for {tarball}: {digest}")
    print(f"tarball ok ({rel['tarball_bytes']} bytes, sha256 verified)")

    traces_dir = ROOT / cfg["paths"]["traces_dir"]
    if not traces_dir.exists():
        print("unpacking")
        with tarfile.open(tarball, "r:xz") as tf:
            try:
                tf.extractall(traces_dir.parent, filter="data")
            except TypeError:  # extraction filters need Python >= 3.10.12
                tf.extractall(traces_dir.parent)

    ref_dir = ROOT / cfg["paths"]["reference_dir"]
    for name, url in rel["reference_files"].items():
        if not (ref_dir / name).exists():
            download(url, ref_dir / name)


def inventory(cfg):
    """One record per trace: enough to assign splits without feature extraction."""
    patterns = cfg["invalid_error_patterns"]
    paper_folders = {}
    for ds, ds_cfg in cfg["datasets"].items():
        folders = ds_cfg["folders"].values() if ds_cfg["split"] == "folders" else [ds_cfg["pool_folder"]]
        for f in folders:
            paper_folders[f] = ds
    traces_dir = ROOT / cfg["paths"]["traces_dir"]
    rows = []
    for path in iter_trace_paths(cfg):
        rel = path.relative_to(traces_dir)
        agent, folder = rel.parts[0], rel.parts[1]
        rows.append(
            {
                "path": str(rel),
                "agent": agent,
                "folder": folder,
                "dataset": paper_folders.get(folder, dataset_base(folder)),
                "in_paper": folder in paper_folders,
                "valid": is_valid_trace(load_trace(path), patterns),
            }
        )
    return pd.DataFrame(rows)


def report(cfg, inv):
    results_dir = ROOT / cfg["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = (
        inv.groupby(["agent", "folder", "dataset", "in_paper"])["valid"]
        .agg(n_traces="count", n_valid="sum")
        .reset_index()
    )
    manifest.to_csv(results_dir / "data_manifest.csv", index=False)

    paper = inv[inv["in_paper"]]
    support = paper[paper["valid"]].pivot_table(index="agent", columns="dataset", values="path", aggfunc="count").fillna(0).astype(int)
    print("\nvalid traces per agent x dataset (paper folders; frames and deepshop are pools, split below):")
    print(support.to_string())

    missing = [(a, d) for d in support.columns for a in support.index if support.loc[a, d] == 0]
    if missing:
        sys.exit(f"empty agent x dataset cells, four-site coverage broken: {missing}")

    paper = paper.copy()
    paper["split"] = split_column(cfg, paper)
    print("\nassigned split sizes vs Table 1 (tasks x 14 agents; deficits are failed episodes):")
    for ds, ds_cfg in cfg["datasets"].items():
        sizes = paper[paper["dataset"] == ds]["split"].value_counts()
        for split in ("train", "val", "test"):
            got, ceiling = int(sizes.get(split, 0)), 14 * ds_cfg["tasks"][split]
            print(f"  {ds:14s} {split:5s} {got:5d} / {ceiling:5d}  ({got / ceiling:.1%})")
            if got > ceiling:
                sys.exit(f"{ds} {split}: more traces than Table 1 allows, split logic broken")
    print("\nall 14 agents cover all four environments")

    extra = inv[~inv["in_paper"]]
    if len(extra):
        print(f"note: release also contains {len(extra)} traces in non-paper folders "
              f"({', '.join(sorted(extra['folder'].unique()))}); not used")


def main():
    cfg = load_config()
    ensure_release(cfg)
    report(cfg, inventory(cfg))


if __name__ == "__main__":
    main()
