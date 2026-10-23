from typing import Any

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from ml_pipeline.tuning_config import TuningConfig


def get_hyperparameter_search_space(
    problem_type: str,
    model_name: str,
) -> dict[str, list[Any]]:
    """Return a bounded, dependency-free parameter search space."""

    selector_space = {
        "feature_selector__percentile": [50, 75, 90, 100]
    }

    common_tree_space = {
        **selector_space,
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", None],
    }

    decision_tree_space = {
        **selector_space,
        "model__max_depth": [None, 3, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", None],
    }

    ada_boost_space = {
        **selector_space,
        "model__n_estimators": [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
    }

    classic_gradient_boosting_space = {
        **selector_space,
        "model__n_estimators": [50, 100, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [1, 2, 3, 5],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__subsample": [0.7, 0.85, 1.0],
    }

    gradient_boosting_space = {
        **selector_space,
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [100, 200, 300],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    if problem_type == "classification":
        spaces = {
            "logistic_regression": {
                **selector_space,
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "ridge_classifier": {
                **selector_space,
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "sgd_classifier": {
                **selector_space,
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__penalty": ["l2", "l1", "elasticnet"],
                "model__l1_ratio": [0.05, 0.15, 0.5, 0.9],
            },
            "linear_svc": {
                **selector_space,
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
            "rbf_svc": {
                **selector_space,
                "model__estimator__C": [0.1, 1.0, 10.0, 100.0],
                "model__estimator__gamma": [
                    "scale",
                    "auto",
                    0.01,
                    0.1,
                    1.0,
                ],
            },
            "knn_classifier": {
                **selector_space,
                "model__n_neighbors": [3, 5, 7, 11, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "gaussian_nb": {
                **selector_space,
                "model__var_smoothing": [
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                ],
            },
            "linear_discriminant_analysis": {
                **selector_space,
                "model__shrinkage": [
                    None,
                    "auto",
                    0.0,
                    0.1,
                    0.5,
                    0.9,
                ],
            },
            "decision_tree": decision_tree_space,
            "random_forest": common_tree_space,
            "extra_trees": common_tree_space,
            "ada_boost": ada_boost_space,
            "gradient_boosting": classic_gradient_boosting_space,
            "hist_gradient_boosting": gradient_boosting_space,
        }

    elif problem_type == "regression":
        spaces = {
            "ridge": {
                **selector_space,
                "model__alpha": [
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                    1_000.0,
                ],
            },
            "lasso": {
                **selector_space,
                "model__alpha": [
                    0.001,
                    0.01,
                    0.1,
                    1.0,
                    10.0,
                    100.0,
                ],
            },
            "elastic_net": {
                **selector_space,
                "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.05, 0.25, 0.5, 0.75, 0.95],
            },
            "sgd_regressor": {
                **selector_space,
                "model__alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
                "model__penalty": ["l2", "l1", "elasticnet"],
                "model__l1_ratio": [0.05, 0.15, 0.5, 0.9],
            },
            "huber_regressor": {
                **selector_space,
                "model__epsilon": [1.1, 1.2, 1.35, 1.5, 2.0],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            },
            "linear_svr": {
                **selector_space,
                "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__epsilon": [0.0, 0.01, 0.1, 0.5, 1.0],
            },
            "rbf_svr": {
                **selector_space,
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__epsilon": [0.01, 0.1, 0.5, 1.0],
                "model__gamma": ["scale", "auto", 0.01, 0.1],
            },
            "knn_regressor": {
                **selector_space,
                "model__n_neighbors": [3, 5, 7, 11, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "decision_tree": decision_tree_space,
            "random_forest": common_tree_space,
            "extra_trees": common_tree_space,
            "ada_boost": ada_boost_space,
            "gradient_boosting": classic_gradient_boosting_space,
            "hist_gradient_boosting": gradient_boosting_space,
        }

    else:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    if model_name not in spaces:
        raise ValueError(
            "No hyperparameter search space is registered for "
            f"'{model_name}' in {problem_type}."
        )

    return spaces[model_name]


def _json_parameter(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()

    return value


def tune_model_pipeline(
    pipeline: Pipeline,
    model_name: str,
    problem_type: str,
    X_train: Any,
    y_train: Any,
    cross_validator: Any,
    primary_scoring: Any,
    primary_metric_direction: str,
    config: TuningConfig,
) -> dict[str, Any]:
    """Tune one complete preprocessing-selection-model pipeline."""

    config.validate()
    parameter_space = get_hyperparameter_search_space(
        problem_type,
        model_name,
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=config.n_iter,
        scoring=primary_scoring,
        n_jobs=config.n_jobs,
        cv=cross_validator,
        refit=True,
        random_state=config.random_state,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    raw_best_score = float(search.best_score_)
    best_score_standard_deviation = float(
        search.cv_results_["std_test_score"][search.best_index_]
    )
    reported_best_score = (
        -raw_best_score
        if primary_metric_direction == "lower"
        else raw_best_score
    )
    best_parameters = {
        name: _json_parameter(value)
        for name, value in search.best_params_.items()
    }

    return {
        "pipeline": search.best_estimator_,
        "best_score": reported_best_score,
        "best_score_standard_deviation": (
            best_score_standard_deviation
        ),
        "best_parameters": best_parameters,
        "evaluated_parameter_sets": len(
            search.cv_results_["params"]
        ),
    }
