from copy import deepcopy
import math
from typing import Any

from ml_pipeline.model_registry import ModelSpec


SUPPORTED_DIRECTIONS = {"higher", "lower"}
CONFIDENCE_LEVELS = {
    0: "limited",
    1: "moderate",
    2: "strong",
}


def _safe_float(
    value: Any,
    decimal_places: int,
) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return round(number, decimal_places)


def _is_better(
    candidate: float,
    reference: float,
    direction: str,
) -> bool:
    if direction == "higher":
        return candidate > reference

    return candidate < reference


def _positive_gap(
    winner_score: float,
    comparison_score: float,
    direction: str,
) -> float:
    if direction == "higher":
        return winner_score - comparison_score

    return comparison_score - winner_score


def _relative_value(
    numerator: float,
    denominator: float | None,
    decimal_places: int,
) -> float | None:
    if denominator is None or denominator == 0:
        return None

    return _safe_float(
        numerator / abs(denominator),
        decimal_places,
    )


def _validate_report(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")

    required_fields = {
        "problem_type",
        "primary_metric",
        "primary_metric_direction",
        "model_recommendation",
        "model_comparisons",
        "hyperparameter_tuning",
        "best_model",
        "holdout_metrics",
        "cross_validation",
        "split",
    }
    missing_fields = sorted(required_fields - set(report))

    if missing_fields:
        raise ValueError(
            "Training report is missing fields required for model "
            f"selection explanation: {missing_fields}"
        )

    if report["primary_metric_direction"] not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            "primary_metric_direction must be 'higher' or 'lower'."
        )


def _model_metadata(
    model_specs: list[ModelSpec],
) -> dict[str, dict[str, Any]]:
    if not model_specs:
        raise ValueError("model_specs must not be empty.")

    metadata = {}

    for spec in model_specs:
        if spec.name in metadata:
            raise ValueError("model_specs contains duplicate names.")

        metadata[spec.name] = {
            "family": spec.family,
            "computational_cost": spec.computational_cost,
            "supports_probabilities": spec.supports_probabilities,
            "handles_class_imbalance": (
                spec.handles_class_imbalance
            ),
            "is_baseline": spec.is_baseline,
        }

    return metadata


def _tuning_by_model(
    tuning_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        candidate["model_name"]: candidate
        for candidate in tuning_report.get("candidates", [])
        if candidate.get("status") == "success"
    }


def _rank_successful_models(
    report: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    decimal_places: int,
) -> list[dict[str, Any]]:
    metric = report["primary_metric"]
    direction = report["primary_metric_direction"]
    tuned_models = _tuning_by_model(
        report["hyperparameter_tuning"]
    )
    ranking = []

    for comparison in report["model_comparisons"]:
        if comparison.get("is_baseline"):
            continue

        if comparison.get("status") != "success":
            continue

        model_name = comparison["model_name"]
        metric_report = comparison.get("metrics", {}).get(metric)

        if not isinstance(metric_report, dict):
            raise ValueError(
                f"Successful model '{model_name}' does not contain "
                f"the primary metric '{metric}'."
            )

        untuned_score = _safe_float(
            metric_report.get("mean"),
            decimal_places,
        )

        if untuned_score is None:
            raise ValueError(
                f"Successful model '{model_name}' has an invalid "
                "primary score."
            )

        effective_score = untuned_score
        effective_standard_deviation = _safe_float(
            metric_report.get("standard_deviation"),
            decimal_places,
        )
        selected_variant = "untuned"
        tuning = tuned_models.get(model_name)

        if tuning is not None:
            tuned_score = _safe_float(
                tuning.get("tuned_primary_score"),
                decimal_places,
            )

            if (
                tuned_score is not None
                and _is_better(
                    tuned_score,
                    effective_score,
                    direction,
                )
            ):
                effective_score = tuned_score
                effective_standard_deviation = _safe_float(
                    tuning.get(
                        "tuned_score_standard_deviation"
                    ),
                    decimal_places,
                )
                selected_variant = "tuned"

        model_metadata = metadata.get(model_name, {})
        ranking.append(
            {
                "rank": None,
                "model_name": model_name,
                "display_name": comparison["display_name"],
                "family": model_metadata.get("family"),
                "computational_cost": model_metadata.get(
                    "computational_cost"
                ),
                "selected_variant": selected_variant,
                "cross_validation_primary_score": (
                    effective_score
                ),
                "cross_validation_standard_deviation": (
                    effective_standard_deviation
                ),
                "untuned_primary_score": untuned_score,
                "tuning_attempted": tuning is not None,
            }
        )

    ranking.sort(
        key=lambda item: item["cross_validation_primary_score"],
        reverse=direction == "higher",
    )

    for index, item in enumerate(ranking, start=1):
        item["rank"] = index

    if not ranking:
        raise ValueError(
            "No successful non-baseline model is available for "
            "selection explanation."
        )

    expected_winner = report["best_model"]["model_name"]

    if ranking[0]["model_name"] != expected_winner:
        raise ValueError(
            "Model ranking is inconsistent with the selected model."
        )

    return ranking


def _baseline_assessment(
    report: dict[str, Any],
    winner_score: float,
    decimal_places: int,
) -> dict[str, Any]:
    metric = report["primary_metric"]
    direction = report["primary_metric_direction"]
    baseline_comparison = next(
        (
            comparison
            for comparison in report["model_comparisons"]
            if comparison.get("is_baseline")
            and comparison.get("status") == "success"
        ),
        None,
    )

    if baseline_comparison is None:
        return {
            "status": "unavailable",
            "model_name": None,
            "display_name": None,
            "cross_validation_primary_score": None,
            "absolute_improvement": None,
            "relative_improvement": None,
            "winner_beats_baseline": None,
        }

    baseline_score = _safe_float(
        baseline_comparison["metrics"][metric]["mean"],
        decimal_places,
    )
    improvement = _positive_gap(
        winner_score,
        baseline_score,
        direction,
    )

    return {
        "status": "complete",
        "model_name": baseline_comparison["model_name"],
        "display_name": baseline_comparison["display_name"],
        "cross_validation_primary_score": baseline_score,
        "absolute_improvement": _safe_float(
            improvement,
            decimal_places,
        ),
        "relative_improvement": _relative_value(
            improvement,
            baseline_score,
            decimal_places,
        ),
        "winner_beats_baseline": improvement > 0,
    }


def _holdout_assessment(
    report: dict[str, Any],
    winner_score: float,
    decimal_places: int,
) -> dict[str, Any]:
    metric = report["primary_metric"]
    direction = report["primary_metric_direction"]
    holdout_score = _safe_float(
        report["holdout_metrics"].get(metric),
        decimal_places,
    )

    if holdout_score is None:
        return {
            "status": "unavailable",
            "evaluation_scope": "untouched_holdout",
            "used_for_model_selection": False,
            "raw_rows_included": False,
            "row_count": report["split"].get("test_rows"),
            "primary_metric": metric,
            "primary_score": None,
            "degradation_from_cross_validation": None,
            "relative_degradation": None,
            "generalization_status": "unavailable",
        }

    degradation = (
        winner_score - holdout_score
        if direction == "higher"
        else holdout_score - winner_score
    )
    relative_degradation = _relative_value(
        degradation,
        winner_score,
        decimal_places,
    )

    if degradation <= 0:
        generalization_status = "holdout_at_least_as_strong"
    elif (
        relative_degradation is not None
        and relative_degradation <= 0.10
    ):
        generalization_status = "consistent"
    elif (
        relative_degradation is not None
        and relative_degradation <= 0.20
    ):
        generalization_status = "some_degradation"
    else:
        generalization_status = "substantial_degradation"

    return {
        "status": "complete",
        "evaluation_scope": "untouched_holdout",
        "used_for_model_selection": False,
        "raw_rows_included": False,
        "row_count": report["split"].get("test_rows"),
        "primary_metric": metric,
        "primary_score": holdout_score,
        "degradation_from_cross_validation": _safe_float(
            degradation,
            decimal_places,
        ),
        "relative_degradation": relative_degradation,
        "generalization_status": generalization_status,
    }


def _alternative_assessments(
    report: dict[str, Any],
    ranking: list[dict[str, Any]],
    decimal_places: int,
) -> list[dict[str, Any]]:
    winner = ranking[0]
    direction = report["primary_metric_direction"]
    alternatives = []

    for item in ranking[1:]:
        gap = _positive_gap(
            winner["cross_validation_primary_score"],
            item["cross_validation_primary_score"],
            direction,
        )
        alternatives.append(
            {
                **item,
                "decision": "not_selected",
                "score_gap_from_winner": _safe_float(
                    gap,
                    decimal_places,
                ),
                "relative_gap_from_winner": _relative_value(
                    gap,
                    winner["cross_validation_primary_score"],
                    decimal_places,
                ),
                "reason": (
                    "lower_cross_validation_score"
                    if direction == "higher"
                    else "higher_cross_validation_error"
                ),
            }
        )

    for comparison in report["model_comparisons"]:
        if comparison.get("is_baseline"):
            continue

        if comparison.get("status") != "failed":
            continue

        alternatives.append(
            {
                "rank": None,
                "model_name": comparison["model_name"],
                "display_name": comparison["display_name"],
                "decision": "not_selected",
                "status": "failed",
                "score_gap_from_winner": None,
                "relative_gap_from_winner": None,
                "reason": "cross_validation_failed",
                "error": comparison.get("error"),
            }
        )

    return alternatives


def _skipped_candidates(
    recommendation: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model_name": decision["model_name"],
            "display_name": decision["display_name"],
            "family": decision.get("family"),
            "computational_cost": decision.get(
                "computational_cost"
            ),
            "eligibility": decision.get("eligibility"),
            "recommendation_score": decision.get(
                "recommendation_score"
            ),
            "reasons": list(decision.get("reasons", [])),
        }
        for decision in recommendation.get("model_decisions", [])
        if decision.get("decision") == "skipped"
    ]


def _confidence_assessment(
    report: dict[str, Any],
    ranking: list[dict[str, Any]],
    baseline: dict[str, Any],
    holdout: dict[str, Any],
    decimal_places: int,
) -> dict[str, Any]:
    confidence_rank = 2
    reasons = []
    warnings = []

    if baseline["winner_beats_baseline"] is True:
        reasons.append("winner_beats_dummy_baseline")
    else:
        confidence_rank = 0
        warnings.append("winner_does_not_beat_verified_dummy_baseline")

    if len(ranking) < 2:
        confidence_rank = 0
        warnings.append("only_one_non_baseline_model_was_compared")
        close_competition = None
    else:
        gap = _positive_gap(
            ranking[0]["cross_validation_primary_score"],
            ranking[1]["cross_validation_primary_score"],
            report["primary_metric_direction"],
        )
        relative_gap = _relative_value(
            gap,
            ranking[0]["cross_validation_primary_score"],
            decimal_places,
        )
        close_competition = (
            gap <= 0.01
            if report["problem_type"] == "classification"
            else relative_gap is not None and relative_gap <= 0.02
        )

        if close_competition:
            confidence_rank = min(confidence_rank, 1)
            warnings.append("winner_and_runner_up_are_close")
        else:
            reasons.append("winner_has_clear_cross_validation_lead")

    used_folds = report["cross_validation"].get("used_folds", 0)

    if used_folds >= 3:
        reasons.append("at_least_three_cross_validation_folds_used")
    else:
        confidence_rank = min(confidence_rank, 1)
        warnings.append("fewer_than_three_cross_validation_folds_used")

    holdout_status = holdout["generalization_status"]

    if holdout_status in {
        "consistent",
        "holdout_at_least_as_strong",
    }:
        reasons.append("untouched_holdout_is_consistent")
    elif holdout_status == "some_degradation":
        confidence_rank = min(confidence_rank, 1)
        warnings.append("holdout_shows_some_degradation")
    else:
        confidence_rank = 0
        warnings.append("holdout_evidence_is_limited_or_degraded")

    if report["split"].get("test_rows", 0) < 20:
        confidence_rank = min(confidence_rank, 1)
        warnings.append("holdout_contains_fewer_than_twenty_rows")

    if report.get("failed_model_count", 0) > 0:
        confidence_rank = min(confidence_rank, 1)
        warnings.append("one_or_more_model_comparisons_failed")
    else:
        reasons.append("all_selected_model_comparisons_succeeded")

    tuning_status = report["hyperparameter_tuning"].get("status")

    if tuning_status in {"partial", "failed_fallback"}:
        confidence_rank = min(confidence_rank, 1)
        warnings.append("hyperparameter_tuning_had_failures")

    return {
        "level": CONFIDENCE_LEVELS[confidence_rank],
        "close_competition": close_competition,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": list(dict.fromkeys(warnings)),
        "interpretation": (
            "Evidence strength describes this model-selection run; "
            "it is not a guarantee of future production performance."
        ),
    }


def generate_model_selection_rationale(
    report: dict[str, Any],
    model_specs: list[ModelSpec],
    *,
    decimal_places: int = 6,
) -> dict[str, Any]:
    """Explain model selection without using or exposing holdout rows."""

    _validate_report(report)

    if not 0 <= decimal_places <= 12:
        raise ValueError("decimal_places must be between 0 and 12.")

    original_report = deepcopy(report)
    metadata = _model_metadata(model_specs)
    ranking = _rank_successful_models(
        report,
        metadata,
        decimal_places,
    )
    winner = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else None
    baseline = _baseline_assessment(
        report,
        winner["cross_validation_primary_score"],
        decimal_places,
    )
    holdout = _holdout_assessment(
        report,
        winner["cross_validation_primary_score"],
        decimal_places,
    )
    alternatives = _alternative_assessments(
        report,
        ranking,
        decimal_places,
    )
    skipped = _skipped_candidates(report["model_recommendation"])
    confidence = _confidence_assessment(
        report,
        ranking,
        baseline,
        holdout,
        decimal_places,
    )

    selected_because = [
        "highest_validated_primary_score_after_bounded_tuning",
        "selected_using_training_cross_validation_only",
    ]

    if baseline["winner_beats_baseline"]:
        selected_because.append("beats_dummy_baseline")

    if winner["selected_variant"] == "tuned":
        selected_because.append("tuned_variant_improved_validation_score")

    if report["model_recommendation"]["selection_mode"] == (
        "automatic_recommendation"
    ):
        selected_because.append(
            "included_by_dataset_aware_model_recommender"
        )
    else:
        selected_because.append("included_by_explicit_user_request")

    winner_report = {
        **winner,
        "selected_because": selected_because,
        "beats_dummy_baseline": baseline[
            "winner_beats_baseline"
        ],
    }

    if runner_up is None:
        runner_up_report = None
    else:
        runner_gap = _positive_gap(
            winner["cross_validation_primary_score"],
            runner_up["cross_validation_primary_score"],
            report["primary_metric_direction"],
        )
        runner_up_report = {
            **runner_up,
            "score_gap_from_winner": _safe_float(
                runner_gap,
                decimal_places,
            ),
            "relative_gap_from_winner": _relative_value(
                runner_gap,
                winner["cross_validation_primary_score"],
                decimal_places,
            ),
        }

    summary = [
        (
            f"{winner['display_name']} ranked first among "
            f"{len(ranking)} successfully evaluated non-baseline "
            f"models using training-only cross-validation."
        ),
        (
            "The untouched holdout was evaluated only after model "
            "selection and did not influence the winner."
        ),
    ]

    if runner_up_report is not None:
        summary.append(
            f"{runner_up_report['display_name']} was the runner-up "
            f"with a primary-score gap of "
            f"{runner_up_report['score_gap_from_winner']}."
        )

    if baseline["winner_beats_baseline"] is True:
        summary.append(
            "The selected model outperformed the verified dummy "
            "baseline."
        )
    else:
        summary.append(
            "The selected model did not establish an improvement "
            "over the dummy baseline."
        )

    return {
        "rationale_status": "complete",
        "explanation_scope": "model_selection",
        "selection_data_scope": (
            "training_cross_validation_and_tuning_only"
        ),
        "holdout_used_for_selection": False,
        "private_holdout_rows_exposed": False,
        "problem_type": report["problem_type"],
        "primary_metric": {
            "name": report["primary_metric"],
            "direction": report["primary_metric_direction"],
        },
        "selection_mode": report["model_recommendation"][
            "selection_mode"
        ],
        "registered_candidate_count": report[
            "model_recommendation"
        ]["registered_non_baseline_model_count"],
        "successfully_ranked_model_count": len(ranking),
        "winner": winner_report,
        "runner_up": runner_up_report,
        "baseline_assessment": baseline,
        "model_ranking": ranking,
        "trained_alternative_assessments": alternatives,
        "skipped_candidate_count": len(skipped),
        "skipped_candidate_assessments": skipped,
        "tuning_assessment": {
            "status": report["hyperparameter_tuning"].get("status"),
            "selected_variant": report[
                "hyperparameter_tuning"
            ].get("selected_variant"),
            "selected_model_name": report[
                "hyperparameter_tuning"
            ].get("selected_model_name"),
            "attempted_model_count": report[
                "hyperparameter_tuning"
            ].get("attempted_model_count"),
            "successful_model_count": report[
                "hyperparameter_tuning"
            ].get("successful_model_count"),
            "failed_model_count": report[
                "hyperparameter_tuning"
            ].get("failed_model_count"),
        },
        "holdout_validation": holdout,
        "decision_confidence": confidence,
        "summary": summary,
        "source_training_report_modified": report != original_report,
    }
