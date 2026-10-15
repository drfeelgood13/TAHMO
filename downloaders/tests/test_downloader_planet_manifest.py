import json
from pathlib import Path

import pytest

from downloaders import downloader_planet
from downloaders.downloader_planet import (
    NoEligibleSceneError,
    ORDERS,
    SEARCH_FILTERS,
    date_search_windows,
    ensure_order,
    find_best_scene,
    find_scene_with_fallback,
    load_areas,
)

BOX = {"west_lon": -1.651377, "south_lat": 7.664563, "east_lon": -1.645764, "north_lat": 7.670644}
INPUTS = Path(__file__).resolve().parents[1] / "inputs"
INPUT_NAMES = sorted(path.name for path in INPUTS.glob("*.json"))
LOCATION_FIELDS = ("id", "location_label", "country", "area", "split", "label_year",
                   "green_season_date", "utm_epsg", "n_fields", "bounding_box")


def test_repository_inputs_are_present():
    assert INPUT_NAMES == [
        "ghana_all_110.json", "ghana_test.json", "ghana_train.json", "ghana_val.json",
        "viedia_all_10.json", "viedia_test.json", "viedia_train.json", "viedia_val.json",
    ]


@pytest.mark.parametrize("name", INPUT_NAMES)
def test_repository_inputs_share_one_format(name):
    data = json.loads((INPUTS / name).read_text(encoding="utf-8"))
    locations = data["locations"]

    assert list(data)[:6] == ["split", "dataset", "coordinate_crs", "coordinate_units", "location_count",
                              *(["subarea_count"] if name.startswith("viedia") else ["locations"])]
    assert data["coordinate_crs"] == "EPSG:4326"
    assert data["location_count"] == len(locations)
    assert [location["id"] for location in locations] == list(range(1, len(locations) + 1))
    for location in locations:
        assert all(field in location for field in LOCATION_FIELDS)
        assert data["split"] in ("all", location["split"])
    if name.startswith("viedia"):
        assert data["subarea_count"] == sum(len(location["subareas"]) for location in locations)


@pytest.mark.parametrize("name", INPUT_NAMES)
def test_repository_inputs_load_with_green_season_dates(name):
    data = json.loads((INPUTS / name).read_text(encoding="utf-8-sig"))
    entries = data["locations"]

    areas = load_areas(INPUTS / name)

    assert [area["label"] for area in areas] == [entry["location_label"] for entry in entries]
    assert [area["target_date"] for area in areas] == [entry["green_season_date"] for entry in entries]
    assert all(area["target_date"].startswith("2021-") for area in areas)
    assert all(area["utm_epsg"] == int(entry["utm_epsg"].split(":")[1]) for area, entry in zip(areas, entries))


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


def test_existing_order_on_a_later_page_is_reused_not_resubmitted(monkeypatch, tmp_path: Path):
    calls = []
    pages = {
        ORDERS: {"orders": [{"id": f"other-{i}", "name": f"other-{i}"} for i in range(20)],
                 "_links": {"next": ORDERS + "?page_marker=2"}},
        ORDERS + "?page_marker=2": {"orders": [{"id": "wanted-id", "name": "planet-A-123"}], "_links": {}},
    }

    def api_json(_http, method, url, **kwargs):
        calls.append((method, url))
        if method != "GET":
            raise AssertionError("must not submit a new order")
        return pages[url]

    monkeypatch.setattr(downloader_planet, "api_json", api_json)
    state = {"request": {"name": "planet-A-123"}}

    assert ensure_order(None, state, tmp_path / "state.json") == "wanted-id"
    assert calls == [("GET", ORDERS), ("GET", ORDERS + "?page_marker=2")]
    assert json.loads((tmp_path / "state.json").read_text())["order_id"] == "wanted-id"


def test_load_areas_reads_green_season_date(tmp_path: Path):
    path = tmp_path / "areas.json"
    path.write_text(json.dumps({"locations": [
        {"location_label": "A", "bounding_box": BOX, "green_season_date": "2021-09-15"},
        {"location_label": "B", "bounding_box": BOX},
    ]}))

    areas = load_areas(path)

    assert [area["target_date"] for area in areas] == ["2021-09-15", None]


def test_load_areas_rejects_bad_green_season_date(tmp_path: Path):
    path = tmp_path / "areas.json"
    path.write_text(json.dumps({"locations": [
        {"location_label": "A", "bounding_box": BOX, "green_season_date": "15-09-2021"},
    ]}))

    with pytest.raises(ValueError, match="green_season_date"):
        load_areas(path)


def test_date_search_windows_are_centred_on_the_date():
    assert date_search_windows("2021-09-15") == (
        ("2021-08-31", "2021-09-30"),
        ("2021-08-01", "2021-10-30"),
        ("2021-06-17", "2021-12-14"),
    )


def test_scene_closest_to_target_date_wins_over_less_cloudy(monkeypatch):
    aoi = {"type": "Polygon", "coordinates": [[(-1.65, 7.66), (-1.64, 7.66), (-1.64, 7.67), (-1.65, 7.67), (-1.65, 7.66)]]}
    features = [
        {"id": id_, "geometry": aoi, "properties": {"acquired": acquired, "cloud_cover": cloud}}
        for id_, acquired, cloud in (
            ("far-clear", "2021-09-01T10:00:00.000000Z", 0.0),
            ("near-cloudy", "2021-09-14T10:00:00.000000Z", 0.08),
            ("later-clear", "2021-09-28T10:00:00.000000Z", 0.0),
        )
    ]
    monkeypatch.setattr(downloader_planet, "api_json", lambda *a, **k: {"features": features, "_links": {}})

    by_date = find_best_scene(None, aoi, "2021-08-31", "2021-09-30", 0.1, 0.999, target="2021-09-15")
    by_cloud = find_best_scene(None, aoi, "2021-08-31", "2021-09-30", 0.1, 0.999)

    assert by_date["id"] == "near-cloudy"
    assert by_cloud["id"] == "far-clear"