import pandas as pd
import numpy as np

def get_shap_explanation(model, race_features, feature_names):
    """Use XGBoost's built-in feature importance as SHAP proxy."""
    try:
        import shap
        n = model.n_features_in_
        feature_cols = feature_names[:n]
        for f in feature_cols:
            if f not in race_features.columns:
                race_features[f] = 0.0
        X = race_features[feature_cols].fillna(0).values
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
                })
        return pd.DataFrame(rows)
    except Exception:
        importance   = model.feature_importances_
        n            = model.n_features_in_
        feature_cols = feature_names[:11]  # only use actual features
        
        X = race_features[feature_cols].fillna(0).values
        # Pad to match model's expected features
        if X.shape[1] < n:
            X = np.hstack([X, np.zeros((X.shape[0], n - X.shape[1]))])
        
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
                })
        return pd.DataFrame(rows)

def get_top_factors(shap_df: pd.DataFrame,
                    driver: str, top_n: int = 6) -> pd.DataFrame:
    d = shap_df[shap_df["driver"] == driver].copy()
    d["abs_shap"] = d["shap_value"].abs()
    return d.nlargest(top_n, "abs_shap").reset_index(drop=True)
