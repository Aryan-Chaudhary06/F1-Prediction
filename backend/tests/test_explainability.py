"""
tests/test_explainability.py
────────────────────────────────
Regression tests for explainability.py.

Two things are being guarded here (see RaceMindAI conversation history):

  1. The approximate (non-SHAP) fallback path used to hardcode
     `feature_names[:11]` — a leftover from when the race model had
     exactly 11 features. After weather (§6.4) and FP2 practice-pace
     (§6.5) features were added, FEATURES grew to 19, and the fallback
     path was silently dropping the last 8 features (all of weather +
     practice pace) from every approximate explanation, with no error.
  2. There was no way to tell whether a given explanation came from real
     SHAP values (shap.TreeExplainer) or the crude approximate fallback —
     the two are NOT mathematically comparable, so mixing them under one
     meaning was a trust issue independent of the truncation bug. Every
     row must now carry an explicit explanation_method label.

Uses a REAL, small XGBClassifier (not mocked) so these tests exercise the
actual shap/no-shap code paths rather than testing against a fake stand-in.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import builtins
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from app.models import explainability as expl


N_FEATURES = 19  # matches the current race model's FEATURES length
# (11 base + 6 weather + 2 practice-pace) — deliberately NOT hardcoded
# anywhere in explainability.py itself; this test just needs SOME number
# bigger than the old stale "11" to prove the fix generalizes.
FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]


def _train_tiny_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, N_FEATURES))
    # Make BOTH the first and the very last feature matter, so a
    # regression back to `[:11]` would visibly lose signal from feat_18.
    y = (X[:, 0] + X[:, N_FEATURES - 1] > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, verbosity=0)
    model.fit(X, y)
    race_features = pd.DataFrame(X, columns=FEATURE_NAMES)
    race_features["driver"] = [f"D{i}" for i in range(200)]
    return model, race_features


def _hide_shap_module(monkeypatch):
    """Forces get_shap_explanation() down the ImportError fallback path,
    regardless of whether `shap` is actually installed in this environment."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("simulated: shap not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


class TestApproximateFallbackCoversAllFeatures:
    """The regression guard for the exact bug found: hardcoded [:11]
    silently dropping every feature added after the 11th."""

    def test_all_features_present_not_truncated_to_eleven(self, monkeypatch):
        _hide_shap_module(monkeypatch)
        model, race_features = _train_tiny_model()

        shap_df = expl.get_shap_explanation(model, race_features, FEATURE_NAMES)

        assert shap_df["feature"].nunique() == N_FEATURES, (
            f"Expected all {N_FEATURES} features in the approximate "
            f"explanation, got {shap_df['feature'].nunique()} — this is "
            f"exactly the truncation bug (hardcoded feature_names[:11]) "
            f"this test exists to catch."
        )

    def test_the_last_feature_specifically_is_not_dropped(self, monkeypatch):
        # The most direct possible check: feat_18 (index 18, the 19th
        # feature) would have been silently missing entirely under the
        # old `feature_names[:11]` bug.
        _hide_shap_module(monkeypatch)
        model, race_features = _train_tiny_model()
        shap_df = expl.get_shap_explanation(model, race_features, FEATURE_NAMES)
        assert "feat_18" in shap_df["feature"].values

    def test_uses_model_n_features_in_not_a_hardcoded_count(self, monkeypatch):
        # Train a DIFFERENT-sized model (7 features, matching e.g. the
        # qualifying model's shape) and confirm the function adapts,
        # rather than assuming any fixed number.
        _hide_shap_module(monkeypatch)
        rng = np.random.default_rng(1)
        small_features = [f"feat_{i}" for i in range(7)]
        X = rng.normal(size=(100, 7))
        y = (X[:, 0] > 0).astype(int)
        small_model = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0)
        small_model.fit(X, y)
        race_features = pd.DataFrame(X, columns=small_features)
        race_features["driver"] = [f"D{i}" for i in range(100)]

        shap_df = expl.get_shap_explanation(small_model, race_features, small_features)
        assert shap_df["feature"].nunique() == 7


class TestExplanationMethodLabeling:
    """Every row must be explicitly tagged with which method produced it —
    the two are not mathematically comparable and must never be silently
    mixed under one meaning."""

    def test_fallback_path_is_labeled_approximate(self, monkeypatch):
        _hide_shap_module(monkeypatch)
        model, race_features = _train_tiny_model()
        shap_df = expl.get_shap_explanation(model, race_features, FEATURE_NAMES)

        assert "explanation_method" in shap_df.columns
        assert (shap_df["explanation_method"] == "approximate").all()

    def test_real_shap_path_is_labeled_shap(self):
        pytest.importorskip("shap", reason="shap not installed in this environment")
        model, race_features = _train_tiny_model()
        shap_df = expl.get_shap_explanation(model, race_features, FEATURE_NAMES)

        assert "explanation_method" in shap_df.columns
        assert (shap_df["explanation_method"] == "shap").all()

    def test_get_top_factors_preserves_the_label(self, monkeypatch):
        # The label must survive get_top_factors()'s nlargest() filtering —
        # a caller reading factors.to_dict() (main.py's actual usage
        # pattern) must still see it on every returned row.
        _hide_shap_module(monkeypatch)
        model, race_features = _train_tiny_model()
        shap_df = expl.get_shap_explanation(model, race_features, FEATURE_NAMES)
        factors = expl.get_top_factors(shap_df, "D0", top_n=6)

        assert "explanation_method" in factors.columns
        assert (factors["explanation_method"] == "approximate").all()
        assert len(factors) == 6


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
