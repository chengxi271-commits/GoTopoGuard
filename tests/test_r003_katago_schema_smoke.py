import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r003_katago_schema_smoke.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r003_katago_schema_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R003KataGoSchemaSmokeTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_build_hand_checked_query_uses_analysis_fields(self):
        query = self.mod.build_hand_checked_query(max_visits=64)

        self.assertEqual(query["id"], "R003_HAND_CHECKED_V64")
        self.assertEqual(query["boardXSize"], 19)
        self.assertEqual(query["boardYSize"], 19)
        self.assertEqual(query["maxVisits"], 64)
        self.assertTrue(query["includePolicy"])
        self.assertTrue(query["includeOwnership"])
        self.assertEqual(query["initialStones"][0], ["B", "D4"])

    def test_validate_response_schema_accepts_required_fields(self):
        response = {
            "id": "R003_HAND_CHECKED_V64",
            "moveInfos": [{"move": "Q16", "visits": 12, "winrate": 0.5, "scoreLead": 1.0, "prior": 0.1, "pv": ["Q16"]}],
            "rootInfo": {"visits": 64, "winrate": 0.5, "scoreLead": 1.0},
            "policy": [0.0] * 362,
            "ownership": [0.0] * 361,
        }

        result = self.mod.validate_response_schema(response, expected_id="R003_HAND_CHECKED_V64")

        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["missing_top_level_fields"], [])
        self.assertEqual(result["missing_move_info_fields"], [])

    def test_validate_response_schema_accepts_policy_pass_entry(self):
        response = {
            "id": "R003_HAND_CHECKED_V64",
            "moveInfos": [{"move": "Q16", "visits": 12, "winrate": 0.5, "scoreLead": 1.0, "prior": 0.1, "pv": ["Q16"]}],
            "rootInfo": {"visits": 64, "winrate": 0.5, "scoreLead": 1.0},
            "policy": [0.0] * 362,
            "ownership": [0.0] * 361,
        }

        result = self.mod.validate_response_schema(response, expected_id="R003_HAND_CHECKED_V64")

        self.assertEqual(result["status"], "valid")

    def test_validate_response_schema_rejects_missing_policy(self):
        response = {
            "id": "R003_HAND_CHECKED_V64",
            "moveInfos": [{"move": "Q16", "visits": 12, "winrate": 0.5, "scoreLead": 1.0, "prior": 0.1, "pv": ["Q16"]}],
            "rootInfo": {"visits": 64, "winrate": 0.5, "scoreLead": 1.0},
            "ownership": [0.0] * 361,
        }

        result = self.mod.validate_response_schema(response, expected_id="R003_HAND_CHECKED_V64")

        self.assertEqual(result["status"], "invalid")
        self.assertIn("policy", result["missing_top_level_fields"])


if __name__ == "__main__":
    unittest.main()
