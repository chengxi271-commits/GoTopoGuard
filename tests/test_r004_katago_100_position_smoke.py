import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r004_katago_100_position_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r004_katago_100_position_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R004KataGo100PositionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_sgf_coord_to_gtp_flips_rows_and_skips_i(self):
        self.assertEqual(self.mod.sgf_coord_to_gtp((0, 0), 19), "A19")
        self.assertEqual(self.mod.sgf_coord_to_gtp((8, 18), 19), "J1")
        self.assertEqual(self.mod.sgf_coord_to_gtp((18, 0), 19), "T19")
        self.assertEqual(self.mod.sgf_coord_to_gtp(None, 19), "pass")

    def test_build_query_from_game_uses_prefix_moves(self):
        game = self.mod.R001.parse_sgf_mainline("(;SZ[19]KM[6.5]RU[Chinese];B[pd];W[dd];B[qp])")

        query = self.mod.build_query_from_game("sample", game, move_prefix_len=2, max_visits=64)

        self.assertEqual(query["id"], "sample_v64")
        self.assertEqual(query["moves"], [["B", "Q16"], ["W", "D16"]])
        self.assertTrue(query["includePolicy"])
        self.assertTrue(query["includeOwnership"])
        self.assertEqual(query["maxVisits"], 64)

    def test_build_query_from_game_includes_setup_stones(self):
        game = self.mod.R001.parse_sgf_mainline("(;SZ[19]AB[aa]AW[ss];W[dd])")

        query = self.mod.build_query_from_game("setup", game, move_prefix_len=1, max_visits=64)

        self.assertEqual(query["initialStones"], [["B", "A19"], ["W", "T1"]])

    def test_katago_komi_filter_rejects_quarter_point_komi(self):
        self.assertFalse(self.mod.is_katago_komi_compatible("3.75"))
        self.assertTrue(self.mod.is_katago_komi_compatible("6.5"))

    def test_summarize_batch_requires_all_queries_valid(self):
        schema_records = [
            {"source_id": "jgdb", "position_id": "a", "visits": 64, "schema_status": "valid", "latency_seconds": 1.0},
            {"source_id": "jgdb", "position_id": "a", "visits": 256, "schema_status": "invalid", "latency_seconds": 1.5},
        ]

        report = self.mod.summarize_batch(schema_records, expected_positions=1, expected_visits=[64, 256])

        self.assertEqual(report["status"], "schema_invalid")
        self.assertEqual(report["position_count"], 1)
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["schema_valid_count"], 1)

    def test_go_attack_site_fetch_retries_until_records(self):
        calls = []

        def fake_fetch(limit):
            calls.append(limit)
            if len(calls) == 1:
                return [], {"status": "site_js_fetch_failed", "errors": ["timeout"]}
            return [("g.sgf", "(;SZ[19];B[aa])")], {"status": "ok", "fetched_count": 1}

        records, summary = self.mod.fetch_go_attack_site_records_with_retries(1, attempts=2, fetch_fn=fake_fetch)

        self.assertEqual(len(records), 1)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["attempts_used"], 2)


if __name__ == "__main__":
    unittest.main()
