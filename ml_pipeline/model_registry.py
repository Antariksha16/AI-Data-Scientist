from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


SUPPORTED_PROBLEM_TYPES = {"classification", "regression"}


@dataclass(frozen=True)
class ModelSpec:
    """Estimator and metadata used during model comparison."""

    name: str
    display_name: str
    estimator: Any
    is_baseline: bool = False
    family: str = "unspecified"
    computational_cost: str = "medium"
    handles_class_imbalance: bool = False
    supports_probabilities: bool = False
    max_recommended_rows: int | None = None
    max_recommended_transformed_features: int | None = None


def _classification_registry(
    random_state: int,
) -> dict[str, ModelSpec]:
    return {
        "dummy_classifier": ModelSpec(
            name="dummy_classifier",
            display_name="Dummy Classifier",
            estimator=DummyClassifier(strategy="prior"),
            is_baseline=True,
            family="baseline",
            computational_cost="low",
            supports_probabilities=True,
        ),
        "logistic_regression": ModelSpec(
            name="logistic_regression",
            display_name="Logistic Regression",
            estimator=LogisticRegression(
                max_iter=2_000,
                class_weight="balanced",
                random_state=random_state,
            ),
            family="linear",
            computational_cost="low",
            handles_class_imbalance=True,
            supports_probabilities=True,
        ),
        "ridge_classifier": ModelSpec(
            name="ridge_classifier",
            display_name="Ridge Classifier",
            estimator=RidgeClassifier(class_weight="balanced"),
            family="linear",
            computational_cost="low",
            handles_class_imbalance=True,
        ),
        "sgd_classifier": ModelSpec(
            name="sgd_classifier",
            display_name="SGD Classifier",
            estimator=SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=2_000,
                tol=1e-3,
                random_state=random_state,
            ),
            family="linear",
            computational_cost="low",
            handles_class_imbalance=True,
            supports_probabilities=True,
        ),
        "linear_svc": ModelSpec(
            name="linear_svc",
            display_name="Linear Support Vector Classifier",
            estimator=LinearSVC(
                class_weight="balanced",
                dual="auto",
                max_iter=5_000,
                random_state=random_state,
            ),
            family="kernel",
            computational_cost="low",
            handles_class_imbalance=True,
        ),
        "rbf_svc": ModelSpec(
            name="rbf_svc",
            display_name="RBF Support Vector Classifier",
            estimator=CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    cache_size=512,
                ),
                cv=3,
                ensemble=False,
                n_jobs=1,
            ),
            family="kernel",
            computational_cost="high",
            handles_class_imbalance=True,
            supports_probabilities=True,
            max_recommended_rows=20_000,
            max_recommended_transformed_features=2_000,
        ),
        "knn_classifier": ModelSpec(
            name="knn_classifier",
            display_name="K-Nearest Neighbors Classifier",
            estimator=KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                n_jobs=1,
            ),
            family="neighbor",
            computational_cost="medium",
            supports_probabilities=True,
            max_recommended_rows=30_000,
            max_recommended_transformed_features=1_000,
        ),
        "gaussian_nb": ModelSpec(
            name="gaussian_nb",
            display_name="Gaussian Naive Bayes",
            estimator=GaussianNB(),
            family="probabilistic",
            computational_cost="low",
            supports_probabilities=True,
        ),
        "linear_discriminant_analysis": ModelSpec(
            name="linear_discriminant_analysis",
            display_name="Linear Discriminant Analysis",
            estimator=LinearDiscriminantAnalysis(
                solver="lsqr",
                shrinkage="auto",
            ),
            family="discriminant",
            computational_cost="medium",
            supports_probabilities=True,
            max_recommended_transformed_features=5_000,
        ),
        "decision_tree": ModelSpec(
            name="decision_tree",
            display_name="Decision Tree Classifier",
            estimator=DecisionTreeClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            family="tree",
            computational_cost="low",
            handles_class_imbalance=True,
            supports_probabilities=True,
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            display_name="Random Forest Classifier",
            estimator=RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced_subsample",
                random_state=random_state,
                n_jobs=1,
            ),
            family="bagging",
            computational_cost="high",
            handles_class_imbalance=True,
            supports_probabilities=True,
            max_recommended_rows=100_000,
        ),
        "extra_trees": ModelSpec(
            name="extra_trees",
            display_name="Extra Trees Classifier",
            estimator=ExtraTreesClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            family="bagging",
            computational_cost="medium",
            handles_class_imbalance=True,
            supports_probabilities=True,
            max_recommended_rows=150_000,
        ),
        "ada_boost": ModelSpec(
            name="ada_boost",
            display_name="AdaBoost Classifier",
            estimator=AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_state,
            ),
            family="boosting",
            computational_cost="medium",
            supports_probabilities=True,
            max_recommended_rows=100_000,
        ),
        "gradient_boosting": ModelSpec(
            name="gradient_boosting",
            display_name="Gradient Boosting Classifier",
            estimator=GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_state,
            ),
            family="boosting",
            computational_cost="high",
            supports_probabilities=True,
            max_recommended_rows=50_000,
            max_recommended_transformed_features=5_000,
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            display_name="Histogram Gradient Boosting Classifier",
            estimator=HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=random_state,
            ),
            family="boosting",
            computational_cost="medium",
            handles_class_imbalance=True,
            supports_probabilities=True,
        ),
    }


def _regression_registry(
    random_state: int,
) -> dict[str, ModelSpec]:
    return {
        "dummy_regressor": ModelSpec(
            name="dummy_regressor",
            display_name="Dummy Regressor",
            estimator=DummyRegressor(strategy="mean"),
            is_baseline=True,
            family="baseline",
            computational_cost="low",
        ),
        "ridge": ModelSpec(
            name="ridge",
            display_name="Ridge Regression",
            estimator=Ridge(),
            family="linear",
            computational_cost="low",
        ),
        "lasso": ModelSpec(
            name="lasso",
            display_name="Lasso Regression",
            estimator=Lasso(
                alpha=1.0,
                max_iter=10_000,
                random_state=random_state,
            ),
            family="linear",
            computational_cost="low",
        ),
        "elastic_net": ModelSpec(
            name="elastic_net",
            display_name="Elastic Net Regression",
            estimator=ElasticNet(
                alpha=1.0,
                l1_ratio=0.5,
                max_iter=10_000,
                random_state=random_state,
            ),
            family="linear",
            computational_cost="low",
        ),
        "sgd_regressor": ModelSpec(
            name="sgd_regressor",
            display_name="SGD Regressor",
            estimator=SGDRegressor(
                loss="squared_error",
                max_iter=2_000,
                tol=1e-3,
                random_state=random_state,
            ),
            family="linear",
            computational_cost="low",
        ),
        "huber_regressor": ModelSpec(
            name="huber_regressor",
            display_name="Huber Regressor",
            estimator=HuberRegressor(max_iter=500),
            family="robust_linear",
            computational_cost="medium",
            max_recommended_rows=50_000,
            max_recommended_transformed_features=5_000,
        ),
        "linear_svr": ModelSpec(
            name="linear_svr",
            display_name="Linear Support Vector Regressor",
            estimator=LinearSVR(
                dual="auto",
                max_iter=5_000,
                random_state=random_state,
            ),
            family="kernel",
            computational_cost="low",
        ),
        "rbf_svr": ModelSpec(
            name="rbf_svr",
            display_name="RBF Support Vector Regressor",
            estimator=SVR(kernel="rbf", cache_size=512),
            family="kernel",
            computational_cost="high",
            max_recommended_rows=20_000,
            max_recommended_transformed_features=2_000,
        ),
        "knn_regressor": ModelSpec(
            name="knn_regressor",
            display_name="K-Nearest Neighbors Regressor",
            estimator=KNeighborsRegressor(
                n_neighbors=5,
                weights="distance",
                n_jobs=1,
            ),
            family="neighbor",
            computational_cost="medium",
            max_recommended_rows=30_000,
            max_recommended_transformed_features=1_000,
        ),
        "decision_tree": ModelSpec(
            name="decision_tree",
            display_name="Decision Tree Regressor",
            estimator=DecisionTreeRegressor(
                random_state=random_state,
            ),
            family="tree",
            computational_cost="low",
        ),
        "random_forest": ModelSpec(
            name="random_forest",
            display_name="Random Forest Regressor",
            estimator=RandomForestRegressor(
                n_estimators=150,
                random_state=random_state,
                n_jobs=1,
            ),
            family="bagging",
            computational_cost="high",
            max_recommended_rows=100_000,
        ),
        "extra_trees": ModelSpec(
            name="extra_trees",
            display_name="Extra Trees Regressor",
            estimator=ExtraTreesRegressor(
                n_estimators=150,
                random_state=random_state,
                n_jobs=1,
            ),
            family="bagging",
            computational_cost="medium",
            max_recommended_rows=150_000,
        ),
        "ada_boost": ModelSpec(
            name="ada_boost",
            display_name="AdaBoost Regressor",
            estimator=AdaBoostRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_state,
            ),
            family="boosting",
            computational_cost="medium",
            max_recommended_rows=100_000,
        ),
        "gradient_boosting": ModelSpec(
            name="gradient_boosting",
            display_name="Gradient Boosting Regressor",
            estimator=GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_state,
            ),
            family="boosting",
            computational_cost="high",
            max_recommended_rows=50_000,
            max_recommended_transformed_features=5_000,
        ),
        "hist_gradient_boosting": ModelSpec(
            name="hist_gradient_boosting",
            display_name="Histogram Gradient Boosting Regressor",
            estimator=HistGradientBoostingRegressor(
                random_state=random_state,
            ),
            family="boosting",
            computational_cost="medium",
        ),
    }


def get_model_registry(
    problem_type: str,
    random_state: int = 42,
    requested_models: list[str] | None = None,
) -> list[ModelSpec]:
    """
    Return deterministic candidate models and one dummy benchmark.

    When requested_models is supplied, the relevant dummy model is
    still included so trained estimators can be compared with a
    no-skill reference.
    """

    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    if problem_type == "classification":
        registry = _classification_registry(random_state)
    else:
        registry = _regression_registry(random_state)

    baseline_name = next(
        name
        for name, spec in registry.items()
        if spec.is_baseline
    )

    if requested_models is None:
        selected_names = list(registry)
    else:
        if not requested_models:
            raise ValueError(
                "requested_models must contain at least one model."
            )

        duplicate_names = sorted(
            {
                name
                for name in requested_models
                if requested_models.count(name) > 1
            }
        )

        if duplicate_names:
            raise ValueError(
                "requested_models contains duplicates: "
                f"{duplicate_names}"
            )

        unknown_names = sorted(
            set(requested_models) - set(registry)
        )

        if unknown_names:
            raise ValueError(
                "Unknown model names for "
                f"{problem_type}: {unknown_names}"
            )

        selected_names = [baseline_name]
        selected_names.extend(
            name
            for name in requested_models
            if name != baseline_name
        )

    if all(registry[name].is_baseline for name in selected_names):
        raise ValueError(
            "At least one non-baseline model must be requested."
        )

    return [registry[name] for name in selected_names]
