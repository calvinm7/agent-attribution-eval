"""Four analyses the paper does not report, all on its own data and protocol.

  metrics   macro F1 vs top-1 vs chance on the same confusion matrices
  transfer  full cross-environment matrix plus pooled-site and cross-site
            cells, checked against the paper's Tables 9/10/14 and the
            classifier artifacts shipped with the trace release
  openset   leave-one-agent-out open-set suite: AUROC, EER, OSCR, macro F1 on
            knowns, miss rate at fixed false-alert rates, grouped into easy
            unknowns (no same-family sibling in training) vs hard unknowns
  scaling   site-scaling curve: hold out one paper environment, train on
            k = 1..4 of the 3 remaining paper environments plus WebGames,
            raw vs fixed-budget, with bootstrap CIs

The openset unknown score is the classifier's maximum class probability, as
in the reference implementation. False alert rate: fraction of known-agent
traces flagged unknown. Miss rate: fraction of unknown-agent traces accepted
as known.
"""

import argparse
import itertools
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

from common import ROOT, all_datasets, load_config, load_features, load_paper_numbers, split_column
from features import FEATURE_SPEC
from train import base_estimator, search_fit

FEATURES = list(FEATURE_SPEC)

ARTIFACT_TAGS = {
    "2wikimultihop": "wiki_ood_all",
    "frames": "frames_ood_all",
    "webshop": "webshop_ood_all",
    "deepshop": "deepshop_ood_all",
}


def load_artifact(cfg, *parts):
    path = (ROOT / cfg["paths"]["traces_dir"] / "classifiers").joinpath(*parts)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def results_dir(cfg):
    d = ROOT / cfg["paths"]["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def sites(cfg):
    out = {}
    for ds, ds_cfg in cfg["datasets"].items():
        out.setdefault(ds_cfg["site"], []).append(ds)
    return out


def sibling(cfg, dataset):
    site = cfg["datasets"][dataset]["site"]
    return next(d for d in cfg["datasets"] if d != dataset and cfg["datasets"][d]["site"] == site)


def fast_macro_f1(y_true, y_pred, n_classes):
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(1) + cm.sum(0)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


# ---------------------------------------------------------------- metrics

def run_metrics(cfg):
    paper = load_paper_numbers()
    rng = np.random.default_rng(0)
    n_perm = cfg["metrics"]["permutation_baseline_n"]
    rows = []
    for dataset in cfg["datasets"]:
        pred = pd.read_csv(results_dir(cfg) / f"predictions_{dataset}.csv")
        enc = LabelEncoder().fit(sorted(cfg["agents"]))
        y, p = enc.transform(pred["agent"]), enc.transform(pred["pred"])
        k = len(enc.classes_)
        per_class = f1_score(y, p, average=None, labels=range(k))
        perm = np.array([fast_macro_f1(y, rng.permutation(p), k) for _ in range(n_perm)])
        rows.append(
            {
                "dataset": dataset,
                "n_test": len(y),
                "top1_accuracy": round(accuracy_score(y, p), 4),
                "macro_f1": round(f1_score(y, p, average="macro"), 4),
                "best_per_agent_f1": round(float(per_class.max()), 4),
                "best_agent": enc.classes_[per_class.argmax()],
                "worst_per_agent_f1": round(float(per_class.min()), 4),
                "worst_agent": enc.classes_[per_class.argmin()],
                "chance_top1": round(1 / k, 4),
                "permuted_macro_f1_mean": round(float(perm.mean()), 4),
                "permuted_macro_f1_sd": round(float(perm.std()), 4),
                "paper_macro_f1": paper["table9_macro_f1_xgb"][dataset] / 100,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(results_dir(cfg) / "metrics_comparison.csv", index=False)
    print(out.to_string(index=False))


# ---------------------------------------------------------------- transfer

def run_transfer(cfg):
    paper = load_paper_numbers()
    df = load_features(cfg)
    df["split"] = split_column(cfg, df)
    models_dir = ROOT / cfg["paths"]["models_dir"]
    enc = LabelEncoder().fit(sorted(cfg["agents"]))
    k = len(enc.classes_)

    def evaluate(model, sub):
        X = sub[FEATURES].to_numpy(dtype=float)
        y = enc.transform(sub["agent"])
        return fast_macro_f1(y, model.predict(X), k), len(sub)

    rows = []
    single = {ds: joblib.load(models_dir / f"{ds}.joblib")["model"] for ds in cfg["datasets"]}
    for src, dst in itertools.product(cfg["datasets"], repeat=2):
        test_split = df[(df["dataset"] == dst) & (df["split"] == "test")]
        f1_t, n_t = evaluate(single[src], test_split)
        rows.append({"src": src, "dst": dst, "convention": "test_split", "macro_f1": round(f1_t, 4), "n_eval": n_t})
        if src != dst:
            pool = df[(df["dataset"] == dst) & df["valid"]]
            f1_p, n_p = evaluate(single[src], pool)
            art = load_artifact(cfg, ARTIFACT_TAGS[src], "results.json")
            art_f1 = art["models"]["XGBoost"]["ood_reports"].get(dst, {}).get("macro avg", {}).get("f1-score") if art else None
            paper_f1 = paper["table10_macro_f1_xgb"].get(f"{src}->{dst}")
            rows.append(
                {
                    "src": src,
                    "dst": dst,
                    "convention": "full_pool",
                    "macro_f1": round(f1_p, 4),
                    "n_eval": n_p,
                    "artifact_macro_f1": round(art_f1, 4) if art_f1 is not None else "",
                    "paper_macro_f1": round(paper_f1 / 100, 4) if paper_f1 is not None else "",
                }
            )

    for site, members in sites(cfg).items():
        train = df[df["dataset"].isin(members) & (df["split"] == "train")]
        X_tr = train[FEATURES].to_numpy(dtype=float)
        y_tr = enc.transform(train["agent"])
        model_path = models_dir / f"pooled_{site}.joblib"
        if model_path.exists():
            model = joblib.load(model_path)["model"]
        else:
            gs = search_fit(cfg, X_tr, y_tr)
            model = gs.best_estimator_
            joblib.dump({"model": model, "best_params": gs.best_params_}, model_path)
        for dst in cfg["datasets"]:
            in_site = dst in members
            test_split = df[(df["dataset"] == dst) & (df["split"] == "test")]
            f1_t, n_t = evaluate(model, test_split)
            rows.append({"src": f"pooled_{site}", "dst": dst, "convention": "test_split", "macro_f1": round(f1_t, 4), "n_eval": n_t})
            if not in_site:
                pool = df[(df["dataset"] == dst) & df["valid"]]
                f1_p, n_p = evaluate(model, pool)
                rows.append({"src": f"pooled_{site}", "dst": dst, "convention": "full_pool", "macro_f1": round(f1_p, 4), "n_eval": n_p})

    out = pd.DataFrame(rows)
    out.to_csv(results_dir(cfg) / "transfer_matrix.csv", index=False)
    print(out.to_string(index=False))


# ---------------------------------------------------------------- openset

def open_set_metrics(known_scores, known_correct, unknown_scores, far_points):
    y = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    s = np.concatenate([known_scores, unknown_scores])
    auroc = roc_auc_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)  # tpr over knowns, fpr = miss over unknowns
    out = {"auroc": auroc}
    for f in far_points:
        idx = np.argmax(tpr >= 1 - f)  # smallest tpr satisfying FAR <= f
        out[f"miss_at_far_{int(f * 100)}pct"] = float(fpr[idx])
    far = 1 - tpr
    cross = np.argmax(far <= fpr)
    out["eer"] = float((far[cross] + fpr[cross]) / 2)

    order = np.argsort(-s)
    is_known = y[order] == 1
    correct = np.concatenate([known_correct, np.zeros(len(unknown_scores), dtype=bool)])[order]
    ccr = np.cumsum(correct & is_known) / max(len(known_scores), 1)
    fpr_u = np.cumsum(~is_known) / max(len(unknown_scores), 1)
    out["oscr"] = float(np.trapezoid(np.concatenate([[0], ccr, [ccr[-1]]]), np.concatenate([[0], fpr_u, [1]])))
    return out


OPEN_SET_ARTIFACT_DIRS = {
    "2wikimultihop": "2wikimultihop_open_set",
    "frames": "frames_open_set",
    "webshop": "webshop_open_set",
    "deepshop": "deepshop_open_set",
}


def unknown_group(cfg, held_out, split_gemini=False):
    """easy = no same-lineage sibling among the 13 training agents."""
    fams = dict(cfg["families"])
    if split_gemini:  # the paper's Table 11 family definition
        fams["gemini_3_1"], fams["gemini_3_flash"] = "gemini-3.1", "gemini-3-flash"
    siblings = [a for a in cfg["agents"] if a != held_out and fams[a] == fams[held_out]]
    return "hard" if siblings else "easy"


def run_openset(cfg):
    df = load_features(cfg)
    out_path = results_dir(cfg) / "open_set.csv"
    done = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        done = set(zip(prev["dataset"], prev["held_out"]))
    far_points = cfg["open_set"]["far_points"]

    for dataset in cfg["datasets"]:
        for held_out in cfg["agents"]:
            if (dataset, held_out) in done:
                continue
            known = [a for a in cfg["agents"] if a != held_out]
            enc = LabelEncoder().fit(sorted(known))
            split = split_column(cfg, df, agents=known)
            in_ds = df["dataset"] == dataset
            is_known = df["agent"].isin(known)
            train = df[in_ds & (split == "train") & is_known]
            test = df[in_ds & (split == "test") & is_known]
            unknown = df[in_ds & (df["agent"] == held_out)]  # all traces, no validity filter, as in the reference

            X_tr, y_tr = train[FEATURES].to_numpy(dtype=float), enc.transform(train["agent"])
            X_te, y_te = test[FEATURES].to_numpy(dtype=float), enc.transform(test["agent"])
            X_un = unknown[FEATURES].to_numpy(dtype=float)

            model = search_fit(cfg, X_tr, y_tr).best_estimator_
            p_known = model.predict_proba(X_te)
            p_unknown = model.predict_proba(X_un)
            pred = p_known.argmax(1)

            m = open_set_metrics(p_known.max(1), pred == y_te, p_unknown.max(1), far_points)
            art = load_artifact(cfg, OPEN_SET_ARTIFACT_DIRS[dataset], f"open_set_loo_{held_out}", "results.json")
            art_os = ((art or {}).get("open_set") or {}).get("XGBoost") or {}
            row = {
                "dataset": dataset,
                "held_out": held_out,
                "family": cfg["families"][held_out],
                "group": unknown_group(cfg, held_out),
                "n_known": len(test),
                "n_unknown": len(unknown),
                "macro_f1_known": round(f1_score(y_te, pred, average="macro"), 4),
                **{k: round(v, 4) for k, v in m.items()},
                "artifact_auroc": round(art_os["auroc"], 4) if art_os else "",
                "artifact_fpr95": round(art_os["fpr95"], 4) if art_os else "",
            }
            header = not out_path.exists()
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=header, index=False)
            print(f"{dataset} holdout={held_out}: auroc={m['auroc']:.3f} "
                  f"(theirs {row['artifact_auroc'] or 'n/a'}), eer={m['eer']:.3f}")

    folds = pd.read_csv(out_path)
    metric_cols = ["macro_f1_known", "auroc", "eer", "oscr"] + [f"miss_at_far_{int(f * 100)}pct" for f in far_points]
    folds["family"] = folds["held_out"].map(cfg["families"])
    folds["group"] = [unknown_group(cfg, a) for a in folds["held_out"]]
    folds["group_table11"] = [unknown_group(cfg, a, split_gemini=True) for a in folds["held_out"]]
    pieces = []
    for gcol in ("group", "group_table11"):
        agg = folds.groupby(["dataset", gcol])[metric_cols].agg(["mean", "std"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reset_index().rename(columns={gcol: "group"})
        agg.insert(0, "grouping", gcol)
        pieces.append(agg)
        pooled = folds.groupby(gcol)[metric_cols].agg(["mean", "std"]).round(4)
        pooled.columns = ["_".join(c) for c in pooled.columns]
        pooled = pooled.reset_index().rename(columns={gcol: "group"})
        pooled.insert(0, "grouping", gcol)
        pooled.insert(1, "dataset", "all")
        pieces.append(pooled)
    summary = pd.concat(pieces, ignore_index=True)
    summary.to_csv(results_dir(cfg) / "open_set_summary.csv", index=False)
    print(summary.to_string(index=False))


# ---------------------------------------------------------------- scaling

def proportional_subsample(train, budget, rng):
    groups = [g for _, g in train.groupby("agent")]
    total = len(train)
    take = [int(np.floor(budget * len(g) / total)) for g in groups]
    remainders = [budget * len(g) / total - t for g, t in zip(groups, take)]
    for i in np.argsort(remainders)[::-1][: budget - sum(take)]:
        take[i] += 1
    picked = [g.iloc[rng.choice(len(g), size=t, replace=False)] for g, t in zip(groups, take)]
    return pd.concat(picked)


def run_scaling(cfg):
    scfg = cfg["scaling"]
    df = load_features(cfg)
    df["split"] = split_column(cfg, df)
    enc = LabelEncoder().fit(sorted(cfg["agents"]))
    k = len(enc.classes_)
    sources = list(all_datasets(cfg))  # four paper environments + webgames
    train_pools = {ds: df[(df["dataset"] == ds) & (df["split"] == "train")] for ds in sources}
    budget = min(len(t) for t in train_pools.values())
    tests = {ds: df[(df["dataset"] == ds) & (df["split"] == "test")] for ds in cfg["datasets"]}

    B = scfg["bootstrap_resamples"]
    boot_idx = {
        ds: np.random.default_rng(i).integers(0, len(tests[ds]), (B, len(tests[ds])))
        for i, ds in enumerate(sorted(tests))
    }

    fit_rows, boot_store = [], {}
    for heldout in cfg["datasets"]:
        others = [d for d in sources if d != heldout]
        X_te = tests[heldout][FEATURES].to_numpy(dtype=float)
        y_te = enc.transform(tests[heldout]["agent"])
        for r in range(1, len(others) + 1):
            for combo in itertools.combinations(others, r):
                pool = pd.concat([train_pools[d] for d in combo])
                for variant in ("raw", "budget"):
                    for seed in scfg["seeds"]:
                        rng = np.random.default_rng(seed)
                        train = proportional_subsample(pool, budget, rng) if variant == "budget" else pool
                        model = base_estimator(cfg, **scfg["fixed_params"])
                        model.set_params(random_state=seed)
                        model.fit(train[FEATURES].to_numpy(dtype=float), enc.transform(train["agent"]))
                        pred = model.predict(X_te)
                        fid = len(fit_rows)
                        fit_rows.append(
                            {
                                "fit_id": fid,
                                "heldout": heldout,
                                "k": r,
                                "combo": "+".join(combo),
                                "variant": variant,
                                "seed": seed,
                                "sibling_in_train": sibling(cfg, heldout) in combo,
                                "n_train": len(train),
                                "macro_f1": round(fast_macro_f1(y_te, pred, k), 4),
                            }
                        )
                        boot_store[fid] = np.array([fast_macro_f1(y_te[b], pred[b], k) for b in boot_idx[heldout]])
                print(f"scaling {heldout} k={r} combo={'+'.join(combo)} done")

    fits = pd.DataFrame(fit_rows)
    fits.to_csv(results_dir(cfg) / "scaling_fits.csv", index=False)

    lo_q, hi_q = (1 - scfg["confidence"]) / 2, 1 - (1 - scfg["confidence"]) / 2

    def cell_ci(sub):
        boot = np.mean([boot_store[f] for f in sub["fit_id"]], axis=0)
        return pd.Series(
            {
                "macro_f1_mean": round(sub["macro_f1"].mean(), 4),
                "n_fits": len(sub),
                "ci_lo": round(float(np.quantile(boot, lo_q)), 4),
                "ci_hi": round(float(np.quantile(boot, hi_q)), 4),
            }
        )

    aggs = []
    for keys in (["heldout", "k", "variant"], ["k", "variant"], ["k", "variant", "sibling_in_train"]):
        agg = fits.groupby(keys).apply(cell_ci, include_groups=False).reset_index()
        agg.insert(0, "level", "+".join(keys))
        aggs.append(agg)
    curve = pd.concat(aggs, ignore_index=True)
    curve.to_csv(results_dir(cfg) / "scaling_curve.csv", index=False)
    print(curve.to_string(index=False))
    plot_scaling(cfg, fits, curve)


def plot_scaling(cfg, fits, curve):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chance = 1 / 14
    themes = {
        "site_scaling.png": {
            "series": {"raw": "#2a78d6", "budget": "#eb6834", "in": "#1baf7a", "out": "#eda100"},
            "ink": "#0b0b0b", "label": "#52514e", "muted": "#898781",
            "grid": "#e1e0d9", "baseline": "#c3c2b7", "bg": "#ffffff",
        },
        "site_scaling_dark.png": {
            "series": {"raw": "#3987e5", "budget": "#d95926", "in": "#199e70", "out": "#c98500"},
            "ink": "#ffffff", "label": "#c3c2b7", "muted": "#898781",
            "grid": "#2c2c2a", "baseline": "#383835", "bg": "#0d1117",
        },
    }
    fig_dir = ROOT / cfg["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fname, t in themes.items():
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=200, sharey=True)
        fig.patch.set_facecolor(t["bg"])
        for ax in axes:
            ax.set_facecolor(t["bg"])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(t["baseline"])
            ax.grid(axis="y", color=t["grid"], linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", colors=t["muted"], labelsize=8.5, length=3, color=t["baseline"])
            ax.tick_params(axis="y", colors=t["muted"], labelsize=8.5, length=0)
            ax.set_xticks([1, 2, 3, 4])
            ax.set_xlabel("training environments (k of 4)", fontsize=9, color=t["muted"])
            ax.axhline(chance, color=t["muted"], linewidth=1, linestyle=(0, (4, 3)))
            ax.set_ylim(0.0, 0.55)
            ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        def draw(ax, sub, color, label):
            ax.fill_between(sub["k"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.15, linewidth=0)
            ax.plot(sub["k"], sub["macro_f1_mean"], color=color, linewidth=2,
                    marker="o", markersize=5.5, markeredgecolor=t["bg"], markeredgewidth=1.2)
            ax.annotate(label, (sub["k"].iloc[-1], sub["macro_f1_mean"].iloc[-1]), xytext=(7, 0),
                        textcoords="offset points", fontsize=8.5, color=t["label"], va="center",
                        annotation_clip=False)

        ax = axes[0]
        for ds in cfg["datasets"]:
            sub = fits[(fits["heldout"] == ds) & (fits["variant"] == "budget")].groupby("k")["macro_f1"].mean()
            ax.plot(sub.index, sub.values, color=t["baseline"], linewidth=1, zorder=1)
        for variant, label in (("raw", "raw (all traces)"), ("budget", "fixed budget")):
            sub = curve[(curve["level"] == "k+variant") & (curve["variant"] == variant)].sort_values("k")
            draw(ax, sub, t["series"][variant], label)
        ax.annotate("chance (1/14)", (1, chance), xytext=(0, -5), textcoords="offset points",
                    fontsize=8, color=t["muted"], va="top")
        ax.set_ylabel("macro F1 on held-out environment", fontsize=9, color=t["muted"])
        ax.set_title("More training sites, all mixes", fontsize=10, color=t["ink"], loc="left", pad=10)

        ax = axes[1]
        for flag, key, label in ((True, "in", "same-site sibling in training"), (False, "out", "cross-site sources only")):
            sub = curve[(curve["level"] == "k+variant+sibling_in_train")
                        & (curve["variant"] == "budget") & (curve["sibling_in_train"] == flag)].sort_values("k")
            draw(ax, sub, t["series"][key], label)
        ax.set_title("Fixed budget, by source composition", fontsize=10, color=t["ink"], loc="left", pad=10)
        ax.set_xlim(0.8, 4.9)

        fig.suptitle("Cross-site attribution vs number of training environments",
                     fontsize=11.5, color=t["ink"], x=0.01, ha="left", fontweight="bold")
        fig.text(0.01, 0.01,
                 "Lines: mean macro F1 over mixes and 5 seeds. Bands: 95% bootstrap CI over test traces. "
                 "Thin gray: per held-out environment at fixed budget.",
                 fontsize=7.5, color=t["muted"])
        fig.tight_layout(rect=(0, 0.05, 1, 0.92))
        fig.savefig(fig_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fig_dir / fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", choices=["metrics", "transfer", "openset", "scaling", "all"], default="all")
    args = ap.parse_args()
    cfg = load_config()
    steps = ["metrics", "transfer", "openset", "scaling"] if args.analysis == "all" else [args.analysis]
    for step in steps:
        {"metrics": run_metrics, "transfer": run_transfer, "openset": run_openset, "scaling": run_scaling}[step](cfg)


if __name__ == "__main__":
    main()
