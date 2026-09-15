from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EDAConfig:
    """
    Central configuration for automated EDA.

    The object is immutable after creation so one EDA run
    cannot silently change its thresholds.
    """

    # Missing-value thresholds
    high_missing_threshold: float = 0.60
    row_missing_warning_threshold: float = 0.50

    # Categorical-feature thresholds
    rare_category_threshold: float = 0.01
    dominant_category_threshold: float = 0.95
    high_cardinality_count_threshold: int = 50
    high_cardinality_ratio_threshold: float = 0.50
    near_constant_threshold: float = 0.99

    # Numerical-distribution thresholds
    strong_skewness_threshold: float = 1.00
    strong_kurtosis_threshold: float = 3.00
    outlier_iqr_multiplier: float = 1.50
    high_outlier_ratio_threshold: float = 0.05

    # Relationship thresholds
    high_correlation_threshold: float = 0.90
    leakage_correlation_threshold: float = 0.995
    weak_relationship_threshold: float = 0.10
    moderate_relationship_threshold: float = 0.30
    strong_relationship_threshold: float = 0.50
    moderate_vif_threshold: float = 5.00
    high_vif_threshold: float = 10.00

    # Classification imbalance thresholds
    mild_imbalance_ratio: float = 0.70
    moderate_imbalance_ratio: float = 0.40
    severe_imbalance_ratio: float = 0.20

    # Performance and visualization safeguards
    max_categories_displayed: int = 20
    max_features_visualized: int = 20
    max_plot_rows: int = 10_000
    max_pairwise_features: int = 30
    max_relationship_categories: int = 100
    minimum_sample_size: int = 20

    # Datetime-inference safeguards
    datetime_parse_threshold: float = 0.90
    datetime_inference_sample_size: int = 500

    # Output consistency
    decimal_places: int = 4
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values immediately."""

        ratio_fields = {
            "high_missing_threshold": (
                self.high_missing_threshold
            ),
            "row_missing_warning_threshold": (
                self.row_missing_warning_threshold
            ),
            "rare_category_threshold": (
                self.rare_category_threshold
            ),
            "dominant_category_threshold": (
                self.dominant_category_threshold
            ),
            "high_cardinality_ratio_threshold": (
                self.high_cardinality_ratio_threshold
            ),
            "near_constant_threshold": (
                self.near_constant_threshold
            ),
            "high_outlier_ratio_threshold": (
                self.high_outlier_ratio_threshold
            ),
            "high_correlation_threshold": (
                self.high_correlation_threshold
            ),
            "leakage_correlation_threshold": (
                self.leakage_correlation_threshold
            ),
            "weak_relationship_threshold": (
                self.weak_relationship_threshold
            ),
            "moderate_relationship_threshold": (
                self.moderate_relationship_threshold
            ),
            "strong_relationship_threshold": (
                self.strong_relationship_threshold
            ),
            "mild_imbalance_ratio": (
                self.mild_imbalance_ratio
            ),
            "moderate_imbalance_ratio": (
                self.moderate_imbalance_ratio
            ),
            "severe_imbalance_ratio": (
                self.severe_imbalance_ratio
            ),
            "datetime_parse_threshold": (
                self.datetime_parse_threshold
            ),
        }

        for field_name, value in ratio_fields.items():
            if not 0 <= value <= 1:
                raise ValueError(
                    f"{field_name} must be between 0 and 1."
                )

        if self.outlier_iqr_multiplier <= 0:
            raise ValueError(
                "outlier_iqr_multiplier must be greater "
                "than zero."
            )

        if self.strong_skewness_threshold < 0:
            raise ValueError(
                "strong_skewness_threshold cannot be negative."
            )

        if self.strong_kurtosis_threshold < 0:
            raise ValueError(
                "strong_kurtosis_threshold cannot be negative."
            )

        if not (
            0
            <= self.severe_imbalance_ratio
            <= self.moderate_imbalance_ratio
            <= self.mild_imbalance_ratio
            <= 1
        ):
            raise ValueError(
                "Imbalance thresholds must satisfy: "
                "severe <= moderate <= mild."
            )

        if not (
            0
            <= self.weak_relationship_threshold
            <= self.moderate_relationship_threshold
            <= self.strong_relationship_threshold
            <= 1
        ):
            raise ValueError(
                "Relationship thresholds must satisfy: "
                "weak <= moderate <= strong."
            )

        if not (
            1
            < self.moderate_vif_threshold
            <= self.high_vif_threshold
        ):
            raise ValueError(
                "VIF thresholds must satisfy: "
                "1 < moderate <= high."
            )

        positive_integer_fields = {
            "high_cardinality_count_threshold": (
                self.high_cardinality_count_threshold
            ),
            "max_categories_displayed": (
                self.max_categories_displayed
            ),
            "max_features_visualized": (
                self.max_features_visualized
            ),
            "max_plot_rows": self.max_plot_rows,
            "max_pairwise_features": (
                self.max_pairwise_features
            ),
            "max_relationship_categories": (
                self.max_relationship_categories
            ),
            "minimum_sample_size": (
                self.minimum_sample_size
            ),
            "datetime_inference_sample_size": (
                self.datetime_inference_sample_size
            ),
        }

        for field_name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero."
                )

        if self.decimal_places < 0:
            raise ValueError(
                "decimal_places cannot be negative."
            )

    def to_dict(self) -> dict:
        """Return a JSON-compatible configuration dictionary."""

        return asdict(self)