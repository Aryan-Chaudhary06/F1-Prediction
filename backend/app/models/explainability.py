import pandas as pd
import numpy as np


def get_shap_explanation(model, race_features, feature_names):
    """
    Returns a per-driver, per-feature explanation dataframe with columns:
    driver, feature, shap_value, direction, explanation_method.

    `explanation_method` is "shap" (real SHAP values from shap.TreeExplainer)
    or "approximate" (a hand-rolled heuristic used only if the `shap`
    package is unavailable or its computation fails). This label is the
    fix for RaceMindAI_Audit_Phases1-5.md Phase 1 finding #9: previously
    there was no way for downstream code (the API response, the UI) to
    tell which one it was looking at, even though the two are NOT
    mathematically comparable — real SHAP values are additive and sum to
    (prediction - baseline); the approximate fallback
    (feature_importance × direction × probability) does not have that
    property and is a rough directional proxy at best. Silently mixing
    them under one meaning was itself a trust/accuracy risk, independent
    of the labeling fix.

    ALSO FIXES A LIVE BUG found while adding the label above: the
    fallback path used to hardcode `feature_names[:11]` — a leftover from
    when the race model had exactly 11 features. Since then, weather
    (§6.4) and FP2 practice-pace (§6.5) features were added, growing
    FEATURES to 19 — meaning the fallback path was SILENTLY DROPPING THE
    LAST 8 FEATURES (all of weather + practice pace) from every
    approximate explanation, with no error or warning. Now uses
    `model.n_features_in_` the same way the real-SHAP path already
    correctly did, instead of a stale hardcoded count.
    """
    n = model.n_features_in_
    feature_cols = feature_names[:n]
    for f in feature_cols:
        if f not in race_features.columns:
            race_features[f] = 0.0
    X = race_features[feature_cols].fillna(0).values

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(X)
        rows = []
        for i, driver in enumerate(race_features["driver"]):
            for j, feat in enumerate(feature_cols):
                rows.append({
                    "driver":     driver,
                    "feature":    feat,
                    "shap_value": round(float(shap_vals.values[i][j]), 4),
                    "direction":  "positive" if shap_vals.values[i][j] > 0 else "negative",
                    "explanation_method": "shap",
                })
        return pd.DataFrame(rows)
    except ImportError:
        print("[explainability] the 'shap' package isn't installed — "
              "falling back to the approximate (non-SHAP) explanation "
              "method. Run `pip install shap` for real, additive SHAP "
              "explanations instead.")
    except Exception as e:
        # Deliberately still a broad catch (SHAP's TreeExplainer can raise
        # a range of model-compatibility errors depending on the XGBoost/
        # shap version pairing) — but unlike before, this is no longer
        # SILENT. Anyone reading logs can now tell "SHAP unavailable" (see
        # explanation_method=="approximate" below) apart from "everything
        # is fine", and see WHY.
        print(f"[explainability] real SHAP computation failed ({e}) — "
              f"falling back to the approximate (non-SHAP) explanation method.")

    # ── Approximate fallback ─────────────────────────────────────────────
    # NOT real SHAP values — a rough directional heuristic
    # (feature_importance × direction × predicted_probability) used only
    # when the real computation above isn't available. Every row is
    # explicitly tagged explanation_method="approximate" so callers never
    # mistake this for the real thing.
    importance = model.feature_importances_
    probs = model.predict_proba(X)[:, 1]
    rows = []
    for i, driver in enumerate(race_features["driver"]):
        prob = probs[i]
        for j, feat in enumerate(feature_cols):
            direction = 1 if prob > 0.5 else -1
            rows.append({
                "driver":     driver,
                "feature":    feat,
                "shap_value": round(float(importance[j] * direction * prob), 4),
                "direction":  "positive" if direction > 0 else "negative",
                "explanation_method": "approximate",
            })
    return pd.DataFrame(rows)


def get_top_factors(shap_df: pd.DataFrame,
                    driver: str, top_n: int = 6) -> pd.DataFrame:
    d = shap_df[shap_df["driver"] == driver].copy()
    d["abs_shap"] = d["shap_value"].abs()
    return d.nlargest(top_n, "abs_shap").reset_index(drop=True)
