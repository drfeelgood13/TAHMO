import json
from pathlib import Path

from downloader_google_maps import (
    format_location_label,
    load_areas,
    zoom_from_target_resolution,
)


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