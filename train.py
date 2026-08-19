"""Closed-set XGBoost pipeline on the published splits, mirroring the paper's
protocol (Appendix A.3): RandomizedSearchCV, 40 draws from the Table 4 space,
3-fold CV on the train split scored by accuracy, final model refit on the
train split only, features unscaled, labels alphabetical.

For each dataset this writes the fitted model, test predictions with class
probabilities, and a summary comparing macro F1 against the paper's Table 9,
plus a control fit using the hyperparameters the reference run selected.
"""

import json

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from common import ROOT, load_config, load_features, load_paper_numbers, split_column
from features import FEATURE_SPEC

FEATURES = list(FEATURE_SPEC)

# Hyperparameters selected by the reference runs (classifiers/*_ood_all/
# results.json in the release), refit here to separate search variance from
# feature variance.
REFERENCE_BEST_PARAMS = {
    "2wikimultihop": {"subsample": 0.7, "reg_lambda": 2.0, "reg_alpha": 0.01, "n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "colsample_bytree": 0.5},
    "frames": {"subsample": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.01, "n_estimators": 500, "max_depth": 3, "learning_rate": 0.05, "colsample_bytree": 0.5},
    "webshop": {"subsample": 0.6, "reg_lambda": 0.5, "reg_alpha": 0.1, "n_estimators": 400, "max_depth": 5, "learning_rate": 0.05, "colsample_bytree": 0.7},
    "deepshop": {"subsample": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.01, "n_estimators": 500, "max_depth": 3, "learning_rate": 0.05, "colsample_bytree": 0.5},
}


def base_estimator(cfg, **overrides):
    xcfg = cfg["xgboost"]
    return XGBClassifier(
        tree_method=xcfg["tree_method"],
        eval_metric=xcfg["eval_metric"],
        random_state=xcfg["seed"],
        n_jobs=xcfg["model_n_jobs"],
        verbosity=0,
        **overrides,
    )


def search_fit(cfg, X_train, y_train):
    xcfg = cfg["xgboost"]
    gs = RandomizedSearchCV(
        base_estimator(cfg),
        xcfg["space"],
        n_iter=xcfg["search"]["n_iter"],
        cv=xcfg["search"]["cv"],
        scoring=xcfg["search"]["scoring"],
        random_state=xcfg["seed"],
        n_jobs=xcfg["search"]["n_jobs"],
        refit=True,
    )
    gs.fit(X_train, y_train)
    return gs


def xy(df, encoder):
    return df[FEATURES].to_numpy(dtype=float), encoder.transform(df["agent"])


def main():
    cfg = load_config()
    paper = load_paper_numbers()
    df = load_features(cfg)
    df["split"] = split_column(cfg, df)

    encoder = LabelEncoder().fit(sorted(cfg["agents"]))
    models_dir = ROOT / cfg["paths"]["models_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for dataset in cfg["datasets"]:
        d = df[df["dataset"] == dataset]
        train, val, test = (d[d["split"] == s] for s in ("train", "val", "test"))
        X_tr, y_tr = xy(train, encoder)
        X_va, y_va = xy(val, encoder)
        X_te, y_te = xy(test, encoder)

        gs = search_fit(cfg, X_tr, y_tr)
        model = gs.best_estimator_
        proba = model.predict_proba(X_te)
        pred = proba.argmax(axis=1)

        ref_model = base_estimator(cfg, **REFERENCE_BEST_PARAMS[dataset]).fit(X_tr, y_tr)
        ref_pred = ref_model.predict(X_te)

        macro = f1_score(y_te, pred, average="macro")
        summary[dataset] = {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "best_params": gs.best_params_,
            "cv_accuracy": round(gs.best_score_, 4),
            "val_macro_f1": round(f1_score(y_va, model.predict(X_va), average="macro"), 4),
            "test_macro_f1": round(macro, 4),
            "test_accuracy": round(accuracy_score(y_te, pred), 4),
            "test_macro_f1_reference_params": round(f1_score(y_te, ref_pred, average="macro"), 4),
            "paper_table9_macro_f1": paper["table9_macro_f1_xgb"][dataset],
            "delta_pp": round(100 * macro - paper["table9_macro_f1_xgb"][dataset], 2),
        }
        print(f"{dataset}: test macro F1 {100 * macro:.2f} "
              f"(paper {paper['table9_macro_f1_xgb'][dataset]}, "
              f"delta {summary[dataset]['delta_pp']:+.2f}pp)")

        joblib.dump({"model": model, "encoder": encoder, "features": FEATURES}, models_dir / f"{dataset}.joblib")
        out = test[["path", "agent", "dataset"]].copy()
        out["pred"] = encoder.inverse_transform(pred)
        for i, cls in enumerate(encoder.classes_):
            out[f"p_{cls}"] = proba[:, i]
        out.to_csv(results_dir / f"predictions_{dataset}.csv", index=False)

    with open(results_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {results_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
