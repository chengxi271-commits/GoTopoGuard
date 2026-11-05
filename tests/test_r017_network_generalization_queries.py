import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r017_network_generalization_queries.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r017_network_generalization_queries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R017NetworkGeneralizationQueryTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_parse_networks_requires_alias_and_path(self):
        parsed = self.mod.parse_networks("b18=/tmp/a.bin.gz,b40=/tmp/b.bin.gz")

        self.assertEqual(parsed, [("b18", "/tmp/a.bin.gz"), ("b40", "/tmp/b.bin.gz")])

    def test_build_network_query_uses_alias_in_query_id(self):
        record = {
            "position_id": "P1",
            "board": ["." * 19 for _ in range(19)],
            "dataset": "D5",
            "family": "cyclic",
            "komi": "6.5",
        }

        query, metadata = self.mod.build_network_query(record, network_alias="b18", visits=1024)

        self.assertEqual(query["id"], "P1__r017_b18_v1024")
        self.assertEqual(metadata["network_alias"], "b18")


if __name__ == "__main__":
    unittest.main()
