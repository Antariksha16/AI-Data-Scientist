import pytest

from backend.app.core.eda_config import EDAConfig


def test_relationship_configuration_defaults():
    config = EDAConfig()

    assert config.weak_relationship_threshold == 0.10
    assert config.moderate_relationship_threshold == 0.30
    assert config.strong_relationship_threshold == 0.50
    assert config.max_relationship_categories == 100
    assert config.moderate_vif_threshold == 5.0
    assert config.high_vif_threshold == 10.0


def test_relationship_configuration_accepts_custom_values():
    config = EDAConfig(
        weak_relationship_threshold=0.15,
        moderate_relationship_threshold=0.35,
        strong_relationship_threshold=0.60,
        max_relationship_categories=75,
        moderate_vif_threshold=4.0,
        high_vif_threshold=8.0,
    )

    assert config.weak_relationship_threshold == 0.15
    assert config.moderate_relationship_threshold == 0.35
    assert config.strong_relationship_threshold == 0.60
    assert config.max_relationship_categories == 75
    assert config.moderate_vif_threshold == 4.0
    assert config.high_vif_threshold == 8.0


def test_relationship_threshold_order_is_validated():
    with pytest.raises(
        ValueError,
        match="weak <= moderate <= strong",
    ):
        EDAConfig(
            weak_relationship_threshold=0.40,
            moderate_relationship_threshold=0.20,
            strong_relationship_threshold=0.50,
        )


def test_relationship_category_limit_must_be_positive():
    with pytest.raises(
        ValueError,
        match="max_relationship_categories",
    ):
        EDAConfig(max_relationship_categories=0)


def test_vif_threshold_order_is_validated():
    with pytest.raises(
        ValueError,
        match="1 < moderate <= high",
    ):
        EDAConfig(
            moderate_vif_threshold=10.0,
            high_vif_threshold=5.0,
        )