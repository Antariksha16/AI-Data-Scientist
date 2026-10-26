from dataclasses import FrozenInstanceError

import pytest

from ml_pipeline.eda_config import EDAConfig


def test_default_eda_configuration():
    config = EDAConfig()

    assert config.high_missing_threshold == 0.60
    assert config.high_correlation_threshold == 0.90
    assert config.max_plot_rows == 10_000
    assert config.random_state == 42


def test_custom_eda_configuration():
    config = EDAConfig(
        high_missing_threshold=0.70,
        high_correlation_threshold=0.85,
        max_plot_rows=5000,
    )

    assert config.high_missing_threshold == 0.70
    assert config.high_correlation_threshold == 0.85
    assert config.max_plot_rows == 5000


def test_configuration_is_immutable():
    config = EDAConfig()

    with pytest.raises(FrozenInstanceError):
        config.max_plot_rows = 500


def test_invalid_ratio_is_rejected():
    with pytest.raises(
        ValueError,
        match="high_missing_threshold",
    ):
        EDAConfig(
            high_missing_threshold=1.20
        )


def test_invalid_imbalance_order_is_rejected():
    with pytest.raises(
        ValueError,
        match="Imbalance thresholds",
    ):
        EDAConfig(
            severe_imbalance_ratio=0.50,
            moderate_imbalance_ratio=0.40,
        )


def test_invalid_positive_integer_is_rejected():
    with pytest.raises(
        ValueError,
        match="max_plot_rows",
    ):
        EDAConfig(
            max_plot_rows=0
        )


def test_configuration_dictionary():
    config = EDAConfig(
        max_plot_rows=5000
    )

    result = config.to_dict()

    assert isinstance(result, dict)
    assert result["max_plot_rows"] == 5000
    assert result["random_state"] == 42