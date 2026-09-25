"""Offline checks: no credentials or network calls."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "download_jimmyjive_planet.py"
spec = importlib.util.spec_from_file_location("downloader", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class DownloadTests(unittest.TestCase):
    def test_matches_actual_jimmyjive(self):
        tree = ast.parse((SCRIPT.parents[1] / "jimmyjive.py").read_text())
        names = {"EARTH_RADIUS_KM", "CENTER_COORDS", "WIDTH_KM"}
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.FunctionDef) and n.name in
                  {"destination_point", "bbox_from_center_and_width"}) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
        namespace = {"math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "jimmyjive_settings", "exec"), namespace)
        self.assertEqual(m.CENTERS, namespace["CENTER_COORDS"])
        self.assertEqual(m.WIDTH_KM, namespace["WIDTH_KM"])
        for lat, lon in m.CENTERS:
            north, south, west, east = namespace["bbox_from_center_and_width"](lat, lon, 2)
            self.assertEqual(m.shape(m.jimmyjive_aoi(lat, lon)).bounds, (west, south, east, north))

    def test_saved_order_never_posts(self):
        with patch.object(m, "api_json") as api:
            self.assertEqual(m.get_or_create(None, {"order_id": "existing"}, None), "existing")
            api.assert_not_called()

    def test_uncertain_submission_does_not_reorder(self):
        with tempfile.TemporaryDirectory() as d:
            state = {"submission_started": True, "request": {"name": "unique"}}
            with patch.object(m, "api_json", return_value={"orders": []}) as api:
                with self.assertRaisesRegex(RuntimeError, "uncertain"):
                    m.get_or_create(None, state, Path(d) / "state.json")
                self.assertEqual([c.args[1] for c in api.call_args_list], ["GET"])

    def test_lost_response_recovers_order_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            state = {"submission_started": True, "request": {"name": "unique"}}
            with patch.object(m, "api_json", return_value={"orders": [{"name": "unique", "id": "recovered"}]}):
                p = Path(d) / "state.json"
                self.assertEqual(m.get_or_create(None, state, p), "recovered")
                self.assertEqual(json.loads(p.read_text())["order_id"], "recovered")

    def test_creation_intent_saved_before_post(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "state.json"
            state = {"request": {"name": "unique"}}
            def api(_s, method, _url, **_kwargs):
                if method == "GET":
                    return {"orders": []}
                self.assertTrue(json.loads(p.read_text())["submission_started"])
                raise RuntimeError("lost response")
            with patch.object(m, "api_json", side_effect=api):
                with self.assertRaisesRegex(RuntimeError, "lost response"):
                    m.get_or_create(None, state, p)

    def test_coverage_is_a_hard_requirement(self):
        aoi = m.jimmyjive_aoi(*m.CENTERS[0])
        page = {"features": [{"id": "bad", "geometry": m.mapping(m.box(0, 0, 1, 1)), "properties": {}}]}
        with patch.object(m, "api_json", return_value=page):
            with self.assertRaisesRegex(RuntimeError, "No eligible"):
                m.search(None, aoi, "2025-04-01", "2025-05-31", 0.1, 0.999)

    def test_checksum_detects_truncation_and_wrong_content(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.tif"
            info = {"size": 4, "digests": {"sha256": m.hashlib.sha256(b"good").hexdigest()}}
            for contents, expected in [(b"go", False), (b"evil", False), (b"good", True)]:
                p.write_bytes(contents)
                self.assertEqual(m.file_ok(p, info), expected)
            with self.assertRaises(RuntimeError):
                m.safe_path(Path(d), "../escape")


if __name__ == "__main__":
    unittest.main()
