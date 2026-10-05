from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from run_ftw_coordinates import select_scenes


def test_select_scenes_retries_empty_seasonal_window():
    scene_selection = Mock(
        side_effect=[
            ValueError("No sentinel scenes within area"),
            ("early-scene", "late-scene"),
        ]
    )
    args = SimpleNamespace(
        cloud_cover_max=20,
        buffer_days=14,
        year=2024,
        verbose=False,
    )

    scenes = select_scenes(scene_selection, [-1.0, 7.0, -0.9, 7.1], args)

    assert scenes == ("early-scene", "late-scene")
    assert scene_selection.call_count == 2
    assert scene_selection.call_args.kwargs["cloud_cover_max"] == 100
    assert scene_selection.call_args.kwargs["buffer_days"] == 30


def test_select_scenes_does_not_retry_unrelated_value_error():
    scene_selection = Mock(side_effect=ValueError("invalid box"))
    args = SimpleNamespace(
        cloud_cover_max=20,
        buffer_days=14,
        year=2024,
        verbose=False,
    )

    with pytest.raises(ValueError, match="invalid box"):
        select_scenes(scene_selection, [-1.0, 7.0, -0.9, 7.1], args)

    assert scene_selection.call_count == 1