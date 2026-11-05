import importlib.util
import pathlib
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r015_normal_budget_queries.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r015_normal_budget_queries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R015NormalBudgetQueryTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_load_normal_position_records_filters_incompatible_komi(self):
        with tempfile.TemporaryDirectory() as tmp:
            d1 = pathlib.Path(tmp) / "d1.jsonl"
            d2 = pathlib.Path(tmp) / "d2.jsonl"
            d1.write_text(
                '{"position_id":"A","board":["."],"komi":"6.5","board_size":1}\n'
                '{"position_id":"B","board":["."],"komi":"6.25","board_size":1}\n',
                encoding="utf-8",
            )
            d2.write_text('{"position_id":"C","board":["."],"komi":"7.5","board_size":1}\n', encoding="utf-8")

            rows, skipped = self.mod.load_normal_position_records({"D1": str(d1), "D2": str(d2)})

            self.assertEqual([row["position_id"] for row in rows], ["A", "C"])
            self.assertEqual(skipped["incompatible_komi"], 1)
            self.assertEqual(rows[0]["dataset"], "D1")
            self.assertEqual(rows[1]["family"], "natural_engine")

    def test_build_budget_query_uses_r015_query_id(self):
        record = {"position_id": "P", "board": ["."], "komi": "6.5", "board_size": 1, "dataset": "D1", "family": "natural"}

        query, metadata = self.mod.build_budget_query(record, visits=1024)

        self.assertEqual(query["id"], "P__r015_v1024")
        self.assertEqual(metadata["query_id"], "P__r015_v1024")
        self.assertEqual(metadata["run_context"], "R015_normal_budget_regression")


if __name__ == "__main__":
    unittest.main()
