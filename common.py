"""Shared loading and split logic for the reproduction.

Splits follow Table 1 of Lugoloobi et al. (arXiv:2605.14786) as realised by the
reference implementation (github.com/KabakaWilliam/known_actions,
trace_analyzer.py): 2wikimultihop and webshop ship as train/val/test
directories with disjoint question pools; frames and deepshop ship as one pool
directory and are split per agent at the episode level by resplit_assignments.
"""

import hashlib
import json
import random
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_paper_numbers():
    with open(ROOT / "paper_numbers.yaml") as f:
        return yaml.safe_load(f)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def all_datasets(cfg):
    """Paper datasets plus extra scaling-only sources (webgames)."""
    return {**cfg["datasets"], **cfg.get("extra_datasets", {})}


def dataset_base(folder_name):
    """'2wikimultihop_train' -> '2wikimultihop', matching the reference
    dataset_name.rsplit('_', 1) convention."""
    return folder_name.rsplit("_", 1)[0]


def iter_trace_paths(cfg):
    """All trace JSON paths under traces_dir for the paper's 14 agents, sorted
    by path string. The reference implementation enumerates sorted(rglob) and
    the frames/deepshop split depends on this order; sorting the relative path
    string reproduces it (identical prefix for every file)."""
    traces_dir = ROOT / cfg["paths"]["traces_dir"]
    agents = set(cfg["agents"])
    paths = []
    for agent_dir in traces_dir.iterdir():
        if agent_dir.name not in agents:
            continue
        paths.extend(p for p in agent_dir.rglob("*.json") if len(p.relative_to(traces_dir).parts) == 4)
    return sorted(paths, key=lambda p: str(p.relative_to(traces_dir)))


def load_trace(path):
    with open(path) as f:
        return json.load(f)


def is_valid_trace(episode, patterns):
    """Reference filter (_is_valid_trace): drop traces whose error matches an
    API-failure pattern, and traces with no DOM events. Task-level failures
    (timeouts, replanning limit) are kept."""
    err = (episode.get("error") or "").lower()
    if any(p in err for p in patterns):
        return False
    return bool((episode.get("dom_trace") or {}).get("events"))


def load_features(cfg):
    df = pd.read_csv(ROOT / cfg["paths"]["features_file"])
    return df.sort_values("path", kind="stable").reset_index(drop=True)


def resplit_assignments(cfg, df, dataset, agents):
    """Episode-level split for the pool datasets (frames, deepshop).

    Mirrors the reference resplit exactly: valid traces of the pool folder,
    grouped by agent in first-encounter path order, one shared Random(seed)
    consumed sequentially, per-agent shuffle, optional per-agent cap, then
    (0.5, 0.25, 0.25) slices with int() floors and the remainder to test.
    The agent set matters: leave-one-agent-out runs exclude the held-out agent
    before splitting, which shifts every later agent's shuffle.

    Returns {path: split}.
    """
    ds_cfg = cfg["datasets"][dataset]
    pool = df[
        (df["dataset"] == dataset)
        & (df["folder"] == ds_cfg["pool_folder"])
        & df["valid"]
        & df["agent"].isin(agents)
    ].sort_values("path", kind="stable")
    by_agent = {}
    for path, agent in zip(pool["path"], pool["agent"]):
        by_agent.setdefault(agent, []).append(path)
    rng = random.Random(cfg["resplit"]["seed"])
    tr_f, va_f, _ = cfg["resplit"]["fracs"]
    cap = ds_cfg["resplit_cap"]
    out = {}
    for agent, items in by_agent.items():
        rng.shuffle(items)
        if cap is not None:
            items = items[:cap]
        n = len(items)
        n_tr, n_va = int(n * tr_f), int(n * va_f)
        for p in items[:n_tr]:
            out[p] = "train"
        for p in items[n_tr : n_tr + n_va]:
            out[p] = "val"
        for p in items[n_tr + n_va :]:
            out[p] = "test"
    return out


def split_column(cfg, df, agents=None):
    """Split assignment for every row: folder datasets by directory suffix,
    pool datasets by resplit_assignments over `agents` (default: all 14).
    Rows outside the split (invalid, capped out, or held-out agent) get ''."""
    agents = list(cfg["agents"]) if agents is None else list(agents)
    col = pd.Series("", index=df.index)
    for dataset, ds_cfg in all_datasets(cfg).items():
        if ds_cfg["split"] == "folders":
            for split, folder in ds_cfg["folders"].items():
                m = (df["folder"] == folder) & df["valid"] & df["agent"].isin(agents)
                col[m] = split
        else:
            assign = resplit_assignments(cfg, df, dataset, agents)
            m = df["dataset"] == dataset
            col[m] = df.loc[m, "path"].map(assign).fillna("")
    return col
