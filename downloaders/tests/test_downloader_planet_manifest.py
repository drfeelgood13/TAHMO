import json
from pathlib import Path

from downloaders.downloader_planet import (
    NoEligibleSceneError,
    SEARCH_FILTERS,
    find_scene_with_fallback,
    load_areas,
)


def test_load_test_set_areas_preserves_bounds_and_labels(tmp_path: Path):
    manifest = {
        "coordinate_crs": "EPSG:4326",
        "locations": [
            {
                "location_label": "GHANA_area012_GH0418403_0p3m",
                "bounding_box": {
                    "west_lon": -1.651377,
                    "south_lat": 7.664563,
                    "east_lon": -1.645764,
                    "north_lat": 7.670644,
                },
            }
        ],
    }
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(manifest))

    areas = load_areas(path)

    assert len(areas) == 1
    assert areas[0]["label"] == "GHANA_area012_GH0418403_0p3m"
    assert areas[0]["folder"] == areas[0]["label"]
    assert areas[0]["utm_epsg"] == 32630
    coordinates = areas[0]["aoi"]["coordinates"][0]
    assert (
        min(point[0] for point in coordinates),
        min(point[1] for point in coordinates),
        max(point[0] for point in coordinates),
        max(point[1] for point in coordinates),
    ) == (
        -1.651377,
        7.664563,
        -1.645764,
        7.670644,
    )


def test_scene_search_falls_back_to_wider_window():
    calls = []

    def search(_http, _aoi, start, end, cloud, coverage):
        calls.append((start, end, cloud, coverage))
        if len(calls) < 4:
            raise NoEligibleSceneError("No candidate")
        return {"id": "fallback-scene"}

    result = find_scene_with_fallback(
        None,
        {},
        (("2025-08-01", "2025-09-30"), ("2025-07-16", "2025-11-30")),
        search,
    )

    assert result["id"] == "fallback-scene"
    assert calls[:3] == [
        ("2025-08-01", "2025-09-30", *filters) for filters in SEARCH_FILTERS
    ]
    assert calls[3] == ("2025-07-16", "2025-11-30", *SEARCH_FILTERS[0])