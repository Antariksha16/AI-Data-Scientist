from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Validated configuration for baseline model comparison."""

    test_size: float = 0.20
    cv_folds: int = 5
    random_state: int = 42
    feature_selection_percentile: int = 75
    minimum_rows: int = 20
    max_transformed_features: int = 10_000
    max_recommended_models: int = 4
    recommendation_small_dataset_rows: int = 5_000
    recommendation_large_dataset_rows: int = 50_000
    recommendation_high_dimensional_features: int = 1_000
    recommendation_extreme_imbalance_ratio: float = 0.10
    decimal_places: int = 6
    n_jobs: int = 1

    def validate(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError(
                "test_size must be greater than 0 and less than 1."
            )

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")

        if not 1 <= self.feature_selection_percentile <= 100:
            raise ValueError(
                "feature_selection_percentile must be between "
                "1 and 100."
            )

        if self.minimum_rows < 10:
            raise ValueError("minimum_rows must be at least 10.")

        if self.max_transformed_features < 1:
            raise ValueError(
                "max_transformed_features must be at least 1."
            )

        if not 1 <= self.max_recommended_models <= 8:
            raise ValueError(
                "max_recommended_models must be between 1 and 8."
            )

        if self.recommendation_small_dataset_rows < 20:
            raise ValueError(
                "recommendation_small_dataset_rows must be at "
                "least 20."
            )

        if (
            self.recommendation_large_dataset_rows
            <= self.recommendation_small_dataset_rows
        ):
            raise ValueError(
                "recommendation_large_dataset_rows must be greater "
                "than recommendation_small_dataset_rows."
            )

        if self.recommendation_high_dimensional_features < 10:
            raise ValueError(
                "recommendation_high_dimensional_features must be "
                "at least 10."
            )

        if not (
            0
            < self.recommendation_extreme_imbalance_ratio
            <= 0.50
        ):
            raise ValueError(
                "recommendation_extreme_imbalance_ratio must be "
                "greater than 0 and at most 0.50."
            )

        if not 0 <= self.decimal_places <= 12:
            raise ValueError(
                "decimal_places must be between 0 and 12."
            )

        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(
                "n_jobs must be -1 or a positive integer."
            )

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
