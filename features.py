"""The 41 behavioural features of Lugoloobi et al. (arXiv:2605.14786),
reimplemented from the Table 8 spec (Appendix A.5), not from the reference
code. FEATURE_SPEC quotes each Table 8 description; extract_features computes
them in the same order. Known divergences from the spec and the reference
code are listed in REPORT.md.

Usage:
  python features.py             build data/features.csv for the four benchmarks
  python features.py --validate  compare against the reference extractor
"""

import argparse
import importlib.util
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from common import ROOT, all_datasets, is_valid_trace, iter_trace_paths, load_config, load_trace

FEATURE_SPEC = {
    # Event volume
    "n_clicks": "Total number of click events",
    "n_scrolls": "Total number of scroll events",
    "n_navigations": "Total number of page navigation events",
    "n_keydowns": "Total number of keydown events",
    "n_focus": "Total number of input/textarea focus events",
    "n_events_total": "Total events across all types",
    "page_count": "Number of distinct pages visited",
    "n_unique_domains": "Number of unique hostnames visited",
    # Global timing
    "total_duration_s": "Wall-clock episode duration (seconds)",
    "t_first_action_ms": "Time from episode start to first event (ms)",
    "mean_iei_ms": "Mean inter-event interval across all events (ms)",
    "std_iei_ms": "Standard deviation of inter-event intervals (ms)",
    "median_iei_ms": "Median inter-event interval (ms)",
    "p10_iei_ms": "10th percentile inter-event interval (ms)",
    "p90_iei_ms": "90th percentile inter-event interval (ms)",
    "iei_trend": "Ratio of mean IEI in the second half of the episode to the first half",
    # Per-type planning latency
    "mean_click_iei_ms": "Mean inter-click interval (ms)",
    "std_click_iei_ms": "Std of inter-click intervals (ms)",
    "mean_nav_iei_ms": "Mean inter-navigation interval, approximating page dwell time (ms)",
    "std_nav_iei_ms": "Std of inter-navigation intervals (ms)",
    "max_page_dwell_ms": "Maximum single-page dwell time (ms)",
    "mean_key_iei_ms": "Mean inter-keydown interval (ms)",
    "std_key_iei_ms": "Std of inter-keydown intervals (ms)",
    # Scroll behavior
    "max_scroll_pct": "Maximum scroll depth reached, as a percentage of page height",
    "mean_scroll_pct": "Mean scroll depth across all scroll events",
    "n_deep_scrolls": "Number of scroll events reaching >60% page depth",
    "scroll_reversals": "Number of direction reversals in the scroll depth sequence",
    # Click spatial distribution
    "click_x_std": "Standard deviation of click x-coordinates (pixels)",
    "click_y_std": "Standard deviation of click y-coordinates (pixels)",
    "click_bbox_area_frac": "Bounding-box area of all click positions as a fraction of the 1280x768 viewport",
    "click_top_frac": "Fraction of clicks in the top quarter of the viewport (y < 192px)",
    "n_link_clicks": "Number of clicks on anchor elements (href present)",
    "link_click_ratio": "n_link_clicks / n_clicks",
    # Navigation strategy
    "popstate_ratio": "Fraction of navigations triggered by popstate (history back)",
    "scroll_to_click_ratio": "n_scrolls / n_clicks",
    "actions_per_page": "n_events_total / page_count",
    "nav_to_click_ratio": "n_navigations / n_clicks",
    "keydowns_per_page": "n_keydowns / page_count",
    "focus_per_page": "n_focus / page_count",
    "structural_key_ratio": "Fraction of keydowns that are structural keys (Enter, Arrow*, Tab, Escape, Backspace, Delete) vs printable characters",
    # Exit behavior
    "mean_exit_scroll_pct": "Mean scroll depth at beforeunload events",
}


def _iei_stats(ts):
    """Successive diffs of a timestamp sequence; fewer than 2 timestamps
    gives an empty array."""
    if len(ts) < 2:
        return np.array([])
    return np.diff(np.asarray(ts, dtype=float))


def _mean(a):
    return float(np.mean(a)) if len(a) else 0.0


def _std(a):
    return float(np.std(a)) if len(a) else 0.0


def extract_features(episode, cfg):
    fcfg = cfg["features"]
    events = (episode.get("dom_trace") or {}).get("events") or []
    events = sorted(events, key=lambda e: e.get("t_episode", e.get("t", 0)))
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    clicks = by_type.get("click", [])
    scrolls = by_type.get("scroll", [])
    navs = by_type.get("navigate", [])
    keydowns = by_type.get("keydown", [])
    focuses = by_type.get("focus", [])
    unloads = by_type.get("beforeunload", [])

    ts = [e.get("t_episode", e.get("t", 0)) for e in events]
    ieis = _iei_stats(ts)
    f = {}

    f["n_clicks"] = len(clicks)
    f["n_scrolls"] = len(scrolls)
    f["n_navigations"] = len(navs)
    f["n_keydowns"] = len(keydowns)
    f["n_focus"] = len(focuses)
    f["n_events_total"] = len(events)
    page_count = (episode.get("dom_trace") or {}).get("pageCount") or len({e.get("url", "") for e in events})
    f["page_count"] = page_count
    f["n_unique_domains"] = len({urlparse(e.get("url", "")).netloc for e in navs if e.get("url")})

    duration_ms = (episode.get("dom_trace") or {}).get("episodeDuration")
    if duration_ms is None:
        duration_ms = (ts[-1] - ts[0]) if len(ts) >= 2 else 0
    f["total_duration_s"] = duration_ms / 1000.0
    f["t_first_action_ms"] = float(ts[0]) if ts else 0.0
    f["mean_iei_ms"] = _mean(ieis)
    f["std_iei_ms"] = _std(ieis)
    f["median_iei_ms"] = float(np.median(ieis)) if len(ieis) else 0.0
    f["p10_iei_ms"] = float(np.percentile(ieis, 10)) if len(ieis) else 0.0
    f["p90_iei_ms"] = float(np.percentile(ieis, 90)) if len(ieis) else 0.0
    if len(ieis) >= 4:
        mid = len(ieis) // 2
        f["iei_trend"] = _mean(ieis[mid:]) / max(_mean(ieis[:mid]), 1.0)
    else:
        f["iei_trend"] = 1.0

    click_ieis = _iei_stats([e.get("t_episode", e.get("t", 0)) for e in clicks])
    nav_ieis = _iei_stats([e.get("t_episode", e.get("t", 0)) for e in navs])
    key_ieis = _iei_stats([e.get("t_episode", e.get("t", 0)) for e in keydowns])
    f["mean_click_iei_ms"] = _mean(click_ieis)
    f["std_click_iei_ms"] = _std(click_ieis)
    f["mean_nav_iei_ms"] = _mean(nav_ieis)
    f["std_nav_iei_ms"] = _std(nav_ieis)
    f["max_page_dwell_ms"] = float(np.max(nav_ieis)) if len(nav_ieis) else 0.0
    f["mean_key_iei_ms"] = _mean(key_ieis)
    f["std_key_iei_ms"] = _std(key_ieis)

    pcts = [e.get("pct") or 0 for e in scrolls]
    f["max_scroll_pct"] = max(pcts, default=0)
    f["mean_scroll_pct"] = _mean(pcts)
    f["n_deep_scrolls"] = sum(1 for p in pcts if p > fcfg["deep_scroll_pct"])
    diffs = np.diff(pcts) if len(pcts) > 1 else np.array([])
    f["scroll_reversals"] = int(np.sum(diffs[:-1] * diffs[1:] < 0)) if len(diffs) > 1 else 0

    xs = [e.get("x", 0) for e in clicks]
    ys = [e.get("y", 0) for e in clicks]
    f["click_x_std"] = _std(xs)
    f["click_y_std"] = _std(ys)
    vw, vh = fcfg["viewport"]["width"], fcfg["viewport"]["height"]
    if len(clicks) >= 2:
        f["click_bbox_area_frac"] = (max(xs) - min(xs)) * (max(ys) - min(ys)) / float(vw * vh)
    else:
        f["click_bbox_area_frac"] = 0.0
    f["click_top_frac"] = sum(1 for y in ys if y < fcfg["top_quarter_px"]) / max(len(ys), 1)
    f["n_link_clicks"] = sum(1 for e in clicks if e.get("href"))
    f["link_click_ratio"] = f["n_link_clicks"] / max(f["n_clicks"], 1)

    f["popstate_ratio"] = sum(1 for e in navs if e.get("trigger") == "popstate") / max(f["n_navigations"], 1)
    f["scroll_to_click_ratio"] = f["n_scrolls"] / max(f["n_clicks"], 1)
    f["actions_per_page"] = f["n_events_total"] / max(page_count, 1)
    f["nav_to_click_ratio"] = f["n_navigations"] / max(f["n_clicks"], 1)
    f["keydowns_per_page"] = f["n_keydowns"] / max(page_count, 1)
    f["focus_per_page"] = f["n_focus"] / max(page_count, 1)
    structural = set(fcfg["structural_keys"])
    f["structural_key_ratio"] = sum(1 for e in keydowns if e.get("key") in structural) / max(f["n_keydowns"], 1)

    f["mean_exit_scroll_pct"] = _mean([e.get("pct") or 0 for e in unloads])

    assert list(f) == list(FEATURE_SPEC), "feature set drifted from Table 8"
    return f


def build(cfg):
    patterns = cfg["invalid_error_patterns"]
    folder_to_ds = {}
    for ds, ds_cfg in all_datasets(cfg).items():
        folders = ds_cfg["folders"].values() if ds_cfg["split"] == "folders" else [ds_cfg["pool_folder"]]
        for folder in folders:
            folder_to_ds[folder] = ds
    traces_dir = ROOT / cfg["paths"]["traces_dir"]
    rows = []
    mismatched_agent_ids = 0
    for path in iter_trace_paths(cfg):
        rel = path.relative_to(traces_dir)
        agent, folder = rel.parts[0], rel.parts[1]
        if folder not in folder_to_ds:
            continue
        episode = load_trace(path)
        meta = episode.get("meta") or {}
        if meta.get("agent_id") != agent:
            mismatched_agent_ids += 1
        row = {
            "path": str(rel),
            "agent": agent,
            "adapter_family": meta.get("model_family", ""),
            "dataset": folder_to_ds[folder],
            "folder": folder,
            "valid": is_valid_trace(episode, patterns),
            "episode_id": meta.get("episode_id", ""),
        }
        row.update(extract_features(episode, cfg))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / cfg["paths"]["features_file"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} traces, {int(df['valid'].sum())} valid")
    if mismatched_agent_ids:
        print(f"warning: {mismatched_agent_ids} traces whose meta.agent_id differs from directory")
    print(df.groupby(["dataset"])["valid"].agg(["count", "sum"]).rename(columns={"count": "traces", "sum": "valid"}))


def _stub_torch():
    """The reference module imports torch for its LSTM; feature extraction
    does not need it, so satisfy the import with inert stand-ins."""
    import types

    if importlib.util.find_spec("torch") is not None:
        return

    class _AnyAttr(types.ModuleType):
        def __getattr__(self, name):
            full = f"{self.__name__}.{name}"
            if full in sys.modules:
                return sys.modules[full]
            if name and name[0].isupper():
                return type(name, (), {})
            return lambda *a, **k: None

    for name in ["torch", "torch.nn", "torch.nn.utils", "torch.nn.utils.rnn", "torch.utils", "torch.utils.data"]:
        sys.modules.setdefault(name, _AnyAttr(name))


def validate(cfg, sample_per_dataset):
    """Run the reference extractor on the same traces and compare per feature.
    The reference file is fetched by fetch_data.py and never committed."""
    ref_path = ROOT / cfg["paths"]["reference_dir"] / "trace_analyzer.py"
    if not ref_path.exists():
        sys.exit("reference trace_analyzer.py missing, run fetch_data.py first")
    _stub_torch()
    spec = importlib.util.spec_from_file_location("reference_trace_analyzer", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    df = pd.read_csv(ROOT / cfg["paths"]["features_file"])
    df = df[df["valid"]]
    rng = np.random.default_rng(0)
    parts = []
    for _, g in df.groupby("dataset"):
        idx = rng.choice(len(g), size=min(sample_per_dataset, len(g)), replace=False)
        parts.append(g.iloc[idx])
    take = pd.concat(parts)
    traces_dir = ROOT / cfg["paths"]["traces_dir"]
    theirs_rows = []
    for rel in take["path"]:
        episode = load_trace(traces_dir / rel)
        theirs_rows.append(ref.extract_features(episode))
    theirs = pd.DataFrame(theirs_rows, index=take.index)

    report = []
    for feat in FEATURE_SPEC:
        if feat not in theirs.columns:
            report.append({"feature": feat, "in_reference": False})
            continue
        a = take[feat].to_numpy(dtype=float)
        b = theirs[feat].to_numpy(dtype=float)
        exact = float(np.mean(np.isclose(a, b, rtol=1e-9, atol=1e-9)))
        r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
        report.append(
            {
                "feature": feat,
                "in_reference": True,
                "exact_match": round(exact, 4),
                "pearson_r": round(r, 6) if r == r else "",
                "max_abs_diff": round(float(np.max(np.abs(a - b))), 6),
            }
        )
    rep = pd.DataFrame(report)
    out = ROOT / cfg["paths"]["results_dir"] / "feature_validation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    print(rep.to_string(index=False))
    print(f"\nwrote {out} ({len(take)} traces)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--sample", type=int, default=500, help="traces per dataset for --validate")
    args = ap.parse_args()
    cfg = load_config()
    if args.validate:
        validate(cfg, args.sample)
    else:
        build(cfg)


if __name__ == "__main__":
    main()
