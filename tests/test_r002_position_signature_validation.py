import importlib.util
import json
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r002_position_signature_validation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r002_position_signature_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R002PositionSignatureValidationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def write_jsonl(self, rows):
        tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".jsonl")
        with tmp:
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
        return pathlib.Path(tmp.name)

    def test_duplicate_signatures_are_grouped_with_names(self):
        path = self.write_jsonl(
            [
                {"source_id": "a", "name": "g1", "status": "valid", "position_signature": "same"},
                {"source_id": "a", "name": "g2", "status": "valid", "position_signature": "same"},
                {"source_id": "b", "name": "g3", "status": "valid", "position_signature": "other"},
            ]
        )

        report = self.mod.validate_signature_records(path, required_sources={})

        self.assertEqual(report["valid_count"], 3)
        self.assertEqual(report["unique_signature_count"], 2)
        self.assertEqual(report["duplicate_signature_count"], 1)
        self.assertEqual(report["duplicate_signatures"][0]["position_signature"], "same")
        self.assertEqual([item["name"] for item in report["duplicate_signatures"][0]["records"]], ["g1", "g2"])

    def test_required_source_counts_are_completion_failures(self):
        path = self.write_jsonl(
            [
                {"source_id": "jgdb", "name": "g1", "status": "valid", "position_signature": "a"},
                {"source_id": "go_attack_site_examples", "name": "g2", "status": "valid", "position_signature": "b"},
                {"source_id": "go_attack_site_examples", "name": "g3", "status": "invalid_replay", "position_signature": None},
            ]
        )

        report = self.mod.validate_signature_records(
            path,
            required_sources={"jgdb": 1, "katago_kata1_2020_12_12": 1, "go_attack_site_examples": 2},
        )

        self.assertEqual(
            report["completion_failures"],
            {
                "katago_kata1_2020_12_12": {"required": 1, "valid": 0},
                "go_attack_site_examples": {"required": 2, "valid": 1},
            },
        )

    def test_no_duplicates_and_required_counts_is_done(self):
        path = self.write_jsonl(
            [
                {"source_id": "jgdb", "name": "g1", "status": "valid", "position_signature": "a"},
                {"source_id": "katago_kata1_2020_12_12", "name": "g2", "status": "valid", "position_signature": "b"},
                {"source_id": "go_attack_site_examples", "name": "g3", "status": "valid", "position_signature": "c"},
            ]
        )

        report = self.mod.validate_signature_records(
            path,
            required_sources={"jgdb": 1, "katago_kata1_2020_12_12": 1, "go_attack_site_examples": 1},
        )

        self.assertEqual(report["status"], "done")
        self.assertEqual(report["duplicate_signature_count"], 0)
        self.assertEqual(report["completion_failures"], {})


if __name__ == "__main__":
    unittest.main()
