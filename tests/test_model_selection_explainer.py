from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from backend.app.pipeline.model_registry import get_model_registry
from backend.app.pipeline.model_selection_explainer import (
    generate_model_selection_rationale,
)
from backend.app.pipeline.training import train_and_compare_models
from backend.app.pipeline.training_config import TrainingConfig


def _classification_dataframe(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(818)
    signal = rng.normal(size=rows)
    target = (signal + rng.normal(scale=0.9, size=rows) > 0).astype(int)
    target[:20] = np.array([0, 1] * 10)

    return pd.DataFrame(
        {
            "Age": rng.integers(18, 70, size=rows),
            "Signal": signal,
            "Region": rng.choice(["North", "South", "West"], rows),
            "Target": target,
        }
    )


def _regression_dataframe(rows: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(919)
    experience = rng.integers(0, 20, size=rows)
    performance = rng.normal(60, 10, size=rows)
    target = (
        30_000
        + experience * 1_500
        + performance * 160
        + rng.normal(0, 4_000, size=rows)
    )

    return pd.DataFrame(
        {
            "Experience": experience,
            "Performance": performance,
            "Department": rng.choice(["IT", "HR", "Sales"], rows),
            "Salary": target,
        }
    )


def _fast_config() -> TrainingConfig:
    return TrainingConfig(
        cv_folds=3,
        feature_selection_percentile=100,
        minimum_rows=20,
        max_transformed_features=2_000,
        max_recommended_models=3,
        n_jobs=1,
    )


def _metric(mean: float, deviation: float = 0.02) -> dict:
    return {
        "balanced_accuracy": {
            "mean": mean,
            "standard_deviation": deviation,
            "fold_scores": [mean - deviation, mean, mean + deviation],
        }
    }


def _fake_classification_report(
    *,
    runner_score: float = 0.70,
    single_model: bool = False,
    tuned: bool = False,
) -> dict:
    comparisons = [
        {
            "model_name": "dummy_classifier",
            "display_name": "Dummy Classifier",
            "is_baseline": True,
            "status": "success",
            "metrics": _metric(0.50, 0.0),
            "error": None,
        },
        {
            "model_name": "logistic_regression",
            "display_name": "Logistic Regression",
            "is_baseline": False,
            "status": "success",
            "metrics": _metric(0.72),
            "error": None,
        },
    ]

    if not single_model:
        comparisons.append(
            {
                "model_name": "random_forest",
                "display_name": "Random Forest Classifier",
                "is_baseline": False,
                "status": "success",
                "metrics": _metric(runner_score),
                "error": None,
            }
        )

    if tuned:
        tuning = {
            "status": "complete",
            "selected_variant": "tuned",
            "selected_model_name": "logistic_regression",
            "attempted_model_count": 1,
            "successful_model_count": 1,
            "failed_model_count": 0,
            "candidates": [
                {
                    "model_name": "logistic_regression",
                    "status": "success",
                    "untuned_primary_score": 0.72,
                    "tuned_primary_score": 0.78,
                    "tuned_score_standard_deviation": 0.01,
                    "improvement": 0.06,
                }
            ],
        }
        best_score = 0.78
        variant = "tuned"
    else:
        tuning = {
            "status": "disabled",
            "selected_variant": "untuned",
            "selected_model_name": "logistic_regression",
            "attempted_model_count": 0,
            "successful_model_count": 0,
            "failed_model_count": 0,
            "candidates": [],
        }
        best_score = 0.72
        variant = "untuned"

    decisions = [
        {
            "model_name": "logistic_regression",
            "display_name": "Logistic Regression",
            "family": "linear",
            "computational_cost": "low",
            "decision": "selected",
            "eligibility": "eligible",
            "recommendation_score": 108.0,
            "reasons": ["stable_regularized_reference_model"],
        },
        {
            "model_name": "random_forest",
            "display_name": "Random Forest Classifier",
            "family": "bagging",
            "computational_cost": "high",
            "decision": "selected" if not single_model else "skipped",
            "eligibility": "eligible",
            "recommendation_score": 94.0,
            "reasons": ["computational_cost_high"],
        },
        {
            "model_name": "rbf_svc",
            "display_name": "RBF Support Vector Classifier",
            "family": "kernel",
            "computational_cost": "high",
            "decision": "skipped",
            "eligibility": "eligible",
            "recommendation_score": 91.0,
            "reasons": ["candidate_limit_reached"],
        },
    ]

    return {
        "problem_type": "classification",
        "primary_metric": "balanced_accuracy",
        "primary_metric_direction": "higher",
        "model_recommendation": {
            "selection_mode": (
                "user_override" if single_model else "automatic_recommendation"
            ),
            "registered_non_baseline_model_count": 14,
            "model_decisions": decisions,
        },
        "model_comparisons": comparisons,
        "hyperparameter_tuning": tuning,
        "best_model": {
            "model_name": "logistic_regression",
            "display_name": "Logistic Regression",
            "cross_validation_primary_score": best_score,
            "selected_variant": variant,
            "beats_dummy_baseline": True,
        },
        "holdout_metrics": {"balanced_accuracy": best_score - 0.02},
        "cross_validation": {"used_folds": 5},
        "split": {"training_rows": 160, "test_rows": 40},
        "failed_model_count": 0,
    }


def test_classification_training_returns_complete_json_safe_rationale():
    result = train_and_compare_models(
        _classification_dataframe(),
        "Target",
        config=_fast_config(),
    )
    report = result["report"]
    rationale = report["model_selection_rationale"]

    assert rationale["rationale_status"] == "complete"
    assert rationale["winner"]["model_name"] == report["best_model"][
        "model_name"
    ]
    assert rationale["runner_up"] is not None
    assert rationale["holdout_used_for_selection"] is False
    assert rationale["private_holdout_rows_exposed"] is False
    assert rationale["holdout_validation"]["raw_rows_included"] is False
    assert rationale["source_training_report_modified"] is False
    assert rationale["decision_confidence"]["level"] in {
        "strong",
        "moderate",
        "limited",
    }
    json.dumps(report, allow_nan=False)


def test_regression_rationale_respects_lower_is_better_metrics():
    result = train_and_compare_models(
        _regression_dataframe(),
        "Salary",
        problem_type="regression",
        config=_fast_config(),
    )
    rationale = result["report"]["model_selection_rationale"]

    assert rationale["primary_metric"] == {
        "name": "rmse",
        "direction": "lower",
    }
    assert rationale["winner"]["cross_validation_primary_score"] <= (
        rationale["runner_up"]["cross_validation_primary_score"]
    )
    assert rationale["runner_up"]["score_gap_from_winner"] >= 0
    assert rationale["baseline_assessment"]["absolute_improvement"] >= 0
    json.dumps(rationale, allow_nan=False)


def test_tuned_variant_and_tuning_uncertainty_are_explained():
    report = _fake_classification_report(tuned=True)
    rationale = generate_model_selection_rationale(
        report,
        get_model_registry("classification"),
    )

    assert rationale["winner"]["selected_variant"] == "tuned"
    assert rationale["winner"]["cross_validation_primary_score"] == 0.78
    assert rationale["winner"][
        "cross_validation_standard_deviation"
    ] == 0.01
    assert "tuned_variant_improved_validation_score" in rationale[
        "winner"
    ]["selected_because"]


def test_explicit_single_model_has_limited_selection_evidence():
    report = _fake_classification_report(single_model=True)
    rationale = generate_model_selection_rationale(
        report,
        get_model_registry("classification"),
    )

    assert rationale["selection_mode"] == "user_override"
    assert rationale["runner_up"] is None
    assert rationale["decision_confidence"]["level"] == "limited"
    assert "only_one_non_baseline_model_was_compared" in rationale[
        "decision_confidence"
    ]["warnings"]


def test_close_runner_up_creates_an_explicit_caution():
    report = _fake_classification_report(runner_score=0.715)
    rationale = generate_model_selection_rationale(
        report,
        get_model_registry("classification"),
    )

    assert rationale["runner_up"]["score_gap_from_winner"] == 0.005
    assert rationale["decision_confidence"]["close_competition"] is True
    assert rationale["decision_confidence"]["level"] == "moderate"
    assert "winner_and_runner_up_are_close" in rationale[
        "decision_confidence"
    ]["warnings"]


def test_skipped_candidate_reasons_are_preserved():
    rationale = generate_model_selection_rationale(
        _fake_classification_report(),
        get_model_registry("classification"),
    )
    skipped = {
        item["model_name"]: item
        for item in rationale["skipped_candidate_assessments"]
    }

    assert skipped["rbf_svc"]["reasons"] == [
        "candidate_limit_reached"
    ]
    assert rationale["skipped_candidate_count"] == 1


def test_rationale_does_not_modify_source_report():
    report = _fake_classification_report()
    original = deepcopy(report)

    rationale = generate_model_selection_rationale(
        report,
        get_model_registry("classification"),
    )

    assert report == original
    assert rationale["source_training_report_modified"] is False


def test_invalid_rationale_inputs_are_rejected():
    with pytest.raises(ValueError, match="missing fields"):
        generate_model_selection_rationale(
            {"problem_type": "classification"},
            get_model_registry("classification"),
        )

    report = _fake_classification_report()
    report["primary_metric_direction"] = "sideways"

    with pytest.raises(ValueError, match="higher.*lower"):
        generate_model_selection_rationale(
            report,
            get_model_registry("classification"),
        )


def test_rationale_summary_is_human_readable_and_complete():
    rationale = generate_model_selection_rationale(
        _fake_classification_report(tuned=True),
        get_model_registry("classification"),
    )

    assert len(rationale["summary"]) >= 4
    assert all(
        isinstance(sentence, str) and sentence.endswith(".")
        for sentence in rationale["summary"]
    )
    assert "Logistic Regression" in rationale["summary"][0]

