import json
from pathlib import Path

import pytest

from downloader_google_maps import (
    format_location_label,
    load_areas,
    zoom_from_target_resolution,
)

INPUTS = Path(__file__).resolve().parents[1] / "inputs"


@pytest.mark.parametrize("name", sorted(path.name for path in INPUTS.glob("*.json")))
def test_repository_inputs_load(name):
    locations = json.loads((INPUTS / name).read_text(encoding="utf-8"))["locations"]

    areas = load_areas(INPUTS / name)

    assert [area["label"] for area in areas] == [location["location_label"] for location in locations]
    assert [area["bbox"] for area in areas] == [
        tuple(location["bounding_box"][key] for key in ("west_lon", "south_lat", "east_lon", "north_lat"))
        for location in locations
    ]


def test_load_test_set_areas_preserves_boxes_and_labels(tmp_path: Path):
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

    assert areas == [{
        "label": "GHANA_area012_GH0418403_0p3m",
        "center": ((7.664563 + 7.670644) / 2, (-1.651377 - 1.645764) / 2),
        "bbox": (-1.651377, 7.664563, -1.645764, 7.670644),
        "width_km": None,
    }]
    assert zoom_from_target_resolution(2) == 17
    assert format_location_label(areas[0]["label"], "2m") == "GHANA_area012_GH0418403_2m"
    assert format_location_label("Ghana_01", "2m") == "Ghana_01_2m"