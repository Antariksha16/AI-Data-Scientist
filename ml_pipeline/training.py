import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import (
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from ml_pipeline.analyzer import analyze_target
from ml_pipeline.hyperparameter_tuning import (
    tune_model_pipeline,
)
from ml_pipeline.model_registry import (
    ModelSpec,
    get_model_registry,
)
from ml_pipeline.model_recommender import (
    describe_user_model_override,
    recommend_models,
)
from ml_pipeline.model_selection_explainer import (
    generate_model_selection_rationale,
)
from ml_pipeline.preprocessing import (
    prepare_dataset_for_modeling,
)
from ml_pipeline.training_config import TrainingConfig
from ml_pipeline.tuning_config import TuningConfig


SUPPORTED_PROBLEM_TYPES = {"classification", "regression"}


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
    candidate_score: float,
    reference_score: float,
    direction: str,
) -> bool:
    if direction == "higher":
        return candidate_score > reference_score

    return candidate_score < reference_score


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    return value


def _resolve_problem_type(
    df: pd.DataFrame,
    target_column: str,
    requested_problem_type: str | None,
) -> tuple[str, str, dict]:
    target_analysis = analyze_target(df, target_column)
    inferred_problem_type = target_analysis["problem_type"]

    if requested_problem_type is None:
        if inferred_problem_type not in SUPPORTED_PROBLEM_TYPES:
            raise ValueError(
                "The target problem type could not be determined."
            )

        return (
            inferred_problem_type,
            "inferred",
            target_analysis,
        )

    if requested_problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    return (
        requested_problem_type,
        "user_override",
        target_analysis,
    )


def _validate_target(
    target: pd.Series,
    problem_type: str,
) -> None:
    if target.nunique(dropna=True) < 2:
        raise ValueError(
            "The selected target must contain at least two "
            "unique values."
        )

    if problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(target.dtype):
            raise ValueError(
                "A regression target must contain numerical values."
            )

        numeric_target = pd.to_numeric(
            target,
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(numeric_target).all():
            raise ValueError(
                "A regression target must contain only finite values."
            )


def _split_raw_dataset(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    stratify_target = None
    stratified = False

    if problem_type == "classification":
        class_counts = df[target_column].value_counts(
            dropna=False
        )
        class_count = len(class_counts)
        minimum_class_count = int(class_counts.min())
        test_rows = math.ceil(len(df) * config.test_size)
        training_rows = len(df) - test_rows

        if minimum_class_count < 2:
            raise ValueError(
                "Every classification class needs at least two "
                "rows for a stratified holdout split."
            )

        if test_rows < class_count or training_rows < class_count:
            raise ValueError(
                "The requested test_size cannot place every class "
                "in both training and test data."
            )

        stratify_target = df[target_column]
        stratified = True

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=True,
        stratify=stratify_target,
    )

    return train_df.copy(), test_df.copy(), stratified


def _build_cross_validator(
    y_train: pd.Series,
    problem_type: str,
    config: TrainingConfig,
) -> tuple[StratifiedKFold | KFold, int]:
    if problem_type == "classification":
        minimum_class_count = int(
            y_train.value_counts(dropna=False).min()
        )
        folds = min(config.cv_folds, minimum_class_count)

        if folds < 2:
            raise ValueError(
                "Training data needs at least two rows per class "
                "for cross-validation."
            )

        return (
            StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=config.random_state,
            ),
            folds,
        )

    maximum_folds_with_two_validation_rows = len(y_train) // 2
    folds = min(
        config.cv_folds,
        maximum_folds_with_two_validation_rows,
    )

    if folds < 2:
        raise ValueError(
            "Regression training data is too small for "
            "cross-validation."
        )

    return (
        KFold(
            n_splits=folds,
            shuffle=True,
            random_state=config.random_state,
        ),
        folds,
    )


def _build_scoring(
    problem_type: str,
) -> tuple[dict[str, Any], str, str, set[str]]:
    if problem_type == "classification":
        scoring = {
            "balanced_accuracy": make_scorer(
                balanced_accuracy_score
            ),
            "accuracy": make_scorer(accuracy_score),
            "f1_weighted": make_scorer(
                f1_score,
                average="weighted",
                zero_division=0,
            ),
        }
        return (
            scoring,
            "balanced_accuracy",
            "higher",
            set(),
        )

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    return scoring, "rmse", "lower", {"rmse", "mae"}


def _build_model_pipeline(
    preprocessor: Any,
    model_spec: ModelSpec,
    problem_type: str,
    config: TrainingConfig,
) -> Pipeline:
    score_function = (
        f_classif
        if problem_type == "classification"
        else f_regression
    )

    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("variance_filter", VarianceThreshold()),
            (
                "feature_selector",
                SelectPercentile(
                    score_func=score_function,
                    percentile=(
                        config.feature_selection_percentile
                    ),
                ),
            ),
            ("model", clone(model_spec.estimator)),
        ]
    )


def _estimate_transformed_features(
    X_train: pd.DataFrame,
    preprocessing_metadata: dict,
) -> int:
    numerical_features = preprocessing_metadata.get(
        "numerical_features",
        [],
    )
    categorical_features = preprocessing_metadata.get(
        "categorical_features",
        [],
    )

    estimated_categorical_features = sum(
        max(
            1,
            int(X_train[column].nunique(dropna=True)),
        )
        for column in categorical_features
    )

    return len(numerical_features) + estimated_categorical_features


def _summarize_cross_validation(
    raw_scores: dict[str, np.ndarray],
    scoring_names: list[str],
    negative_metrics: set[str],
    decimal_places: int,
) -> tuple[dict, dict[str, float]]:
    metric_report = {}
    unrounded_means = {}

    for metric_name in scoring_names:
        fold_scores = np.asarray(
            raw_scores[f"test_{metric_name}"],
            dtype=float,
        )

        if metric_name in negative_metrics:
            fold_scores = -fold_scores

        mean_score = float(np.mean(fold_scores))
        standard_deviation = float(np.std(fold_scores))
        unrounded_means[metric_name] = mean_score

        metric_report[metric_name] = {
            "mean": _safe_float(mean_score, decimal_places),
            "standard_deviation": _safe_float(
                standard_deviation,
                decimal_places,
            ),
            "fold_scores": [
                _safe_float(score, decimal_places)
                for score in fold_scores
            ],
        }

    return metric_report, unrounded_means


def _evaluate_holdout(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    decimal_places: int,
) -> dict:
    predictions = pipeline.predict(X_test)

    if problem_type == "classification":
        labels = list(pd.unique(y_test))
        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        return {
            "accuracy": _safe_float(
                accuracy_score(y_test, predictions),
                decimal_places,
            ),
            "balanced_accuracy": _safe_float(
                balanced_accuracy_score(y_test, predictions),
                decimal_places,
            ),
            "precision_weighted": _safe_float(
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                decimal_places,
            ),
            "recall_weighted": _safe_float(
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                decimal_places,
            ),
            "f1_weighted": _safe_float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                decimal_places,
            ),
            "class_labels": [
                _json_scalar(label)
                for label in labels
            ],
            "confusion_matrix": matrix.astype(int).tolist(),
        }

    return {
        "rmse": _safe_float(
            root_mean_squared_error(y_test, predictions),
            decimal_places,
        ),
        "mae": _safe_float(
            mean_absolute_error(y_test, predictions),
            decimal_places,
        ),
        "r2": _safe_float(
            r2_score(y_test, predictions),
            decimal_places,
        ),
    }


def _selected_feature_names(
    pipeline: Pipeline,
) -> tuple[list[str], int, int]:
    preprocessor = pipeline.named_steps["preprocessor"]
    variance_filter = pipeline.named_steps["variance_filter"]
    feature_selector = pipeline.named_steps["feature_selector"]

    transformed_names = np.asarray(
        preprocessor.get_feature_names_out(),
        dtype=object,
    )
    after_variance = transformed_names[
        variance_filter.get_support()
    ]
    selected_names = after_variance[
        feature_selector.get_support()
    ]

    return (
        [str(name) for name in selected_names.tolist()],
        len(transformed_names),
        len(after_variance),
    )


def train_and_compare_models(
    df: pd.DataFrame,
    target_column: str,
    *,
    problem_type: str | None = None,
    config: TrainingConfig | None = None,
    tuning_config: TuningConfig | None = None,
    requested_models: list[str] | None = None,
    allow_potential_leakage: bool = False,
) -> dict[str, Any]:
    """
    Compare baseline tabular models and evaluate the best model.

    Raw rows are split before cleaning, outlier, preprocessing, or
    feature-selection decisions. The returned fitted pipeline contains
    preprocessing, variance filtering, supervised feature selection,
    and the winning estimator.
    """

    config = config or TrainingConfig()
    config.validate()

    if tuning_config is not None:
        tuning_config.validate()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.columns.duplicated().any():
        duplicate_columns = sorted(
            set(df.columns[df.columns.duplicated()].tolist())
        )
        raise ValueError(
            f"Dataset contains duplicate columns: {duplicate_columns}"
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    source_df = df.copy(deep=True)
    target_missing_rows = int(
        source_df[target_column].isna().sum()
    )
    usable_df = source_df.loc[
        source_df[target_column].notna()
    ].copy()

    if len(usable_df) < config.minimum_rows:
        raise ValueError(
            "At least "
            f"{config.minimum_rows} rows with non-missing targets "
            "are required for training."
        )

    (
        resolved_problem_type,
        problem_type_source,
        target_analysis,
    ) = _resolve_problem_type(
        usable_df,
        target_column,
        problem_type,
    )

    _validate_target(
        usable_df[target_column],
        resolved_problem_type,
    )

    train_df, test_df, stratified = _split_raw_dataset(
        usable_df,
        target_column,
        resolved_problem_type,
        config,
    )

    prepared = prepare_dataset_for_modeling(
        train_df,
        target_column,
        allow_potential_leakage=allow_potential_leakage,
    )
    X_train = prepared["X"]
    y_train = prepared["y"]
    preprocessor = prepared["preprocessor"]
    preprocessing_report = prepared["report"]
    X_test = test_df.drop(columns=[target_column]).copy()
    y_test = test_df[target_column].copy()

    cross_validator, cv_folds_used = _build_cross_validator(
        y_train,
        resolved_problem_type,
        config,
    )
    scoring, primary_metric, direction, negative_metrics = (
        _build_scoring(resolved_problem_type)
    )

    preprocessing_metadata = preprocessing_report[
        "preprocessing_metadata"
    ]
    estimated_transformed_features = (
        _estimate_transformed_features(
            X_train,
            preprocessing_metadata,
        )
    )

    if (
        estimated_transformed_features
        > config.max_transformed_features
    ):
        raise ValueError(
            "Estimated transformed feature count "
            f"({estimated_transformed_features}) exceeds the "
            "configured safety limit "
            f"({config.max_transformed_features})."
        )

    all_model_specs = get_model_registry(
        resolved_problem_type,
        random_state=config.random_state,
        requested_models=None,
    )

    if requested_models is None:
        model_recommendation = recommend_models(
            all_model_specs,
            X_train,
            y_train,
            resolved_problem_type,
            preprocessing_metadata,
            estimated_transformed_features,
            max_models=config.max_recommended_models,
            small_dataset_rows=(
                config.recommendation_small_dataset_rows
            ),
            large_dataset_rows=(
                config.recommendation_large_dataset_rows
            ),
            high_dimensional_features=(
                config.recommendation_high_dimensional_features
            ),
            extreme_imbalance_ratio=(
                config.recommendation_extreme_imbalance_ratio
            ),
        )
        model_specs = get_model_registry(
            resolved_problem_type,
            random_state=config.random_state,
            requested_models=model_recommendation[
                "selected_models"
            ],
        )
    else:
        model_specs = get_model_registry(
            resolved_problem_type,
            random_state=config.random_state,
            requested_models=requested_models,
        )
        model_recommendation = describe_user_model_override(
            all_model_specs,
            model_specs,
            X_train,
            y_train,
            resolved_problem_type,
            preprocessing_metadata,
            estimated_transformed_features,
            small_dataset_rows=(
                config.recommendation_small_dataset_rows
            ),
            large_dataset_rows=(
                config.recommendation_large_dataset_rows
            ),
            high_dimensional_features=(
                config.recommendation_high_dimensional_features
            ),
            extreme_imbalance_ratio=(
                config.recommendation_extreme_imbalance_ratio
            ),
        )

    comparisons = []
    successful_models: list[
        tuple[ModelSpec, Pipeline, dict[str, float]]
    ] = []

    for model_spec in model_specs:
        pipeline = _build_model_pipeline(
            preprocessor,
            model_spec,
            resolved_problem_type,
            config,
        )

        try:
            raw_scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                scoring=scoring,
                cv=cross_validator,
                n_jobs=config.n_jobs,
                return_train_score=False,
                error_score="raise",
            )
            metric_report, unrounded_means = (
                _summarize_cross_validation(
                    raw_scores,
                    list(scoring),
                    negative_metrics,
                    config.decimal_places,
                )
            )
            comparisons.append(
                {
                    "model_name": model_spec.name,
                    "display_name": model_spec.display_name,
                    "is_baseline": model_spec.is_baseline,
                    "status": "success",
                    "metrics": metric_report,
                    "error": None,
                }
            )
            successful_models.append(
                (model_spec, pipeline, unrounded_means)
            )

        except Exception as error:
            comparisons.append(
                {
                    "model_name": model_spec.name,
                    "display_name": model_spec.display_name,
                    "is_baseline": model_spec.is_baseline,
                    "status": "failed",
                    "metrics": {},
                    "error": str(error),
                }
            )

    eligible_models = [
        item
        for item in successful_models
        if not item[0].is_baseline
    ]

    if not eligible_models:
        raise RuntimeError(
            "Every non-baseline model failed during "
            "cross-validation."
        )

    reverse = direction == "higher"
    eligible_models.sort(
        key=lambda item: item[2][primary_metric],
        reverse=reverse,
    )
    best_spec, best_pipeline, best_cv_means = eligible_models[0]
    best_primary_score = best_cv_means[primary_metric]
    untuned_best_primary_score = best_primary_score
    final_pipeline_is_fitted = False
    selected_variant = "untuned"
    tuning_candidates_report = []

    if tuning_config is None:
        tuning_report = {
            "status": "disabled",
            "configuration": None,
            "attempted_model_count": 0,
            "successful_model_count": 0,
            "failed_model_count": 0,
            "selected_variant": selected_variant,
            "selected_model_name": best_spec.name,
            "candidates": [],
        }

    else:
        candidates_to_tune = eligible_models[
            : tuning_config.top_models
        ]

        for (
            candidate_spec,
            candidate_pipeline,
            candidate_cv_means,
        ) in candidates_to_tune:
            untuned_score = candidate_cv_means[primary_metric]

            try:
                tuning_result = tune_model_pipeline(
                    pipeline=candidate_pipeline,
                    model_name=candidate_spec.name,
                    problem_type=resolved_problem_type,
                    X_train=X_train,
                    y_train=y_train,
                    cross_validator=cross_validator,
                    primary_scoring=scoring[primary_metric],
                    primary_metric_direction=direction,
                    config=tuning_config,
                )
                tuned_score = tuning_result["best_score"]
                improvement = (
                    tuned_score - untuned_score
                    if direction == "higher"
                    else untuned_score - tuned_score
                )

                tuning_candidates_report.append(
                    {
                        "model_name": candidate_spec.name,
                        "display_name": (
                            candidate_spec.display_name
                        ),
                        "status": "success",
                        "untuned_primary_score": _safe_float(
                            untuned_score,
                            config.decimal_places,
                        ),
                        "tuned_primary_score": _safe_float(
                            tuned_score,
                            config.decimal_places,
                        ),
                        "tuned_score_standard_deviation": (
                            _safe_float(
                                tuning_result[
                                    "best_score_standard_deviation"
                                ],
                                config.decimal_places,
                            )
                        ),
                        "improvement": _safe_float(
                            improvement,
                            config.decimal_places,
                        ),
                        "best_parameters": tuning_result[
                            "best_parameters"
                        ],
                        "evaluated_parameter_sets": (
                            tuning_result[
                                "evaluated_parameter_sets"
                            ]
                        ),
                        "error": None,
                    }
                )

                if _is_better(
                    tuned_score,
                    best_primary_score,
                    direction,
                ):
                    best_spec = candidate_spec
                    best_pipeline = tuning_result["pipeline"]
                    best_primary_score = tuned_score
                    untuned_best_primary_score = untuned_score
                    final_pipeline_is_fitted = True
                    selected_variant = "tuned"

            except Exception as error:
                tuning_candidates_report.append(
                    {
                        "model_name": candidate_spec.name,
                        "display_name": (
                            candidate_spec.display_name
                        ),
                        "status": "failed",
                        "untuned_primary_score": _safe_float(
                            untuned_score,
                            config.decimal_places,
                        ),
                        "tuned_primary_score": None,
                        "tuned_score_standard_deviation": None,
                        "improvement": None,
                        "best_parameters": {},
                        "evaluated_parameter_sets": 0,
                        "error": str(error),
                    }
                )

        successful_tuning_count = sum(
            item["status"] == "success"
            for item in tuning_candidates_report
        )
        failed_tuning_count = (
            len(tuning_candidates_report)
            - successful_tuning_count
        )

        if successful_tuning_count == 0:
            tuning_status = "failed_fallback"
        elif failed_tuning_count > 0:
            tuning_status = "partial"
        else:
            tuning_status = "complete"

        tuning_report = {
            "status": tuning_status,
            "configuration": tuning_config.to_dict(),
            "attempted_model_count": len(
                tuning_candidates_report
            ),
            "successful_model_count": successful_tuning_count,
            "failed_model_count": failed_tuning_count,
            "selected_variant": selected_variant,
            "selected_model_name": best_spec.name,
            "candidates": tuning_candidates_report,
        }

    if not final_pipeline_is_fitted:
        best_pipeline.fit(X_train, y_train)

    holdout_metrics = _evaluate_holdout(
        best_pipeline,
        X_test,
        y_test,
        resolved_problem_type,
        config.decimal_places,
    )
    (
        selected_feature_names,
        transformed_feature_count,
        post_variance_feature_count,
    ) = _selected_feature_names(best_pipeline)

    baseline_result = next(
        (
            item
            for item in successful_models
            if item[0].is_baseline
        ),
        None,
    )
    baseline_primary_score = (
        baseline_result[2][primary_metric]
        if baseline_result is not None
        else None
    )
    if baseline_primary_score is None:
        beats_baseline = None
    else:
        beats_baseline = _is_better(
            best_primary_score,
            baseline_primary_score,
            direction,
        )

    failed_model_count = sum(
        comparison["status"] == "failed"
        for comparison in comparisons
    )
    tuning_has_failures = (
        tuning_config is not None
        and tuning_report["failed_model_count"] > 0
    )

    report = {
        "training_status": (
            "complete"
            if (
                failed_model_count == 0
                and not tuning_has_failures
            )
            else "partial"
        ),
        "target_column": target_column,
        "problem_type": resolved_problem_type,
        "problem_type_source": problem_type_source,
        "inferred_problem_type": target_analysis[
            "problem_type"
        ],
        "primary_metric": primary_metric,
        "primary_metric_direction": direction,
        "configuration": config.to_dict(),
        "split": {
            "original_rows": len(source_df),
            "usable_rows": len(usable_df),
            "missing_target_rows_removed": target_missing_rows,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "test_size": config.test_size,
            "random_state": config.random_state,
            "stratified": stratified,
            "preprocessing_decision_scope": (
                "training_split_only"
            ),
        },
        "cross_validation": {
            "requested_folds": config.cv_folds,
            "used_folds": cv_folds_used,
            "splitter": (
                "StratifiedKFold"
                if resolved_problem_type == "classification"
                else "KFold"
            ),
            "shuffle": True,
        },
        "feature_selection": {
            "method": (
                "SelectPercentile(f_classif)"
                if resolved_problem_type == "classification"
                else "SelectPercentile(f_regression)"
            ),
            "percentile": (
                config.feature_selection_percentile
            ),
            "estimated_transformed_features": (
                estimated_transformed_features
            ),
            "transformed_feature_count": (
                transformed_feature_count
            ),
            "post_variance_feature_count": (
                post_variance_feature_count
            ),
            "selected_feature_count": len(
                selected_feature_names
            ),
            "selected_feature_names": selected_feature_names,
        },
        "model_recommendation": model_recommendation,
        "model_comparisons": comparisons,
        "hyperparameter_tuning": tuning_report,
        "successful_model_count": len(successful_models),
        "failed_model_count": failed_model_count,
        "best_model": {
            "model_name": best_spec.name,
            "display_name": best_spec.display_name,
            "cross_validation_primary_score": _safe_float(
                best_primary_score,
                config.decimal_places,
            ),
            "untuned_cross_validation_primary_score": (
                _safe_float(
                    untuned_best_primary_score,
                    config.decimal_places,
                )
            ),
            "selected_variant": selected_variant,
            "tuned": selected_variant == "tuned",
            "baseline_primary_score": _safe_float(
                baseline_primary_score,
                config.decimal_places,
            ),
            "beats_dummy_baseline": beats_baseline,
        },
        "holdout_metrics": holdout_metrics,
        "preprocessing_report": preprocessing_report,
        "automatic_test_data_modification_performed": False,
        "source_dataframe_modified": not df.equals(source_df),
    }

    report["model_selection_rationale"] = (
        generate_model_selection_rationale(
            report,
            all_model_specs,
            decimal_places=config.decimal_places,
        )
    )

    return {
        "pipeline": best_pipeline,
        "report": report,
        "evaluation_context": {
            "X_holdout": X_test.copy(deep=True),
            "y_holdout": y_test.copy(deep=True),
            "evaluation_scope": "untouched_holdout",
            "used_for_model_selection": False,
        },
    }
