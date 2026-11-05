import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r010_topology_stress_generation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r010_topology_stress_generation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R010TopologyStressGenerationTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_cyclic_enclosure_template_is_face_valid(self):
        record = self.mod.generate_position("cyclic_enclosure", seed=0, run_id="R010", split_id="D5_TEST")

        self.assertEqual(len(record["board"]), 19)
        self.assertTrue(record["face_validity"]["legal_board"])
        self.assertTrue(record["face_validity"]["family_face_valid"])
        self.assertGreaterEqual(record["features"]["enclosed_empty_region_count"], 1)

    def test_generate_balanced_positions_per_family(self):
        records, summary = self.mod.generate_stress_split(per_family_count=2, run_id="R010", split_id="D5_TEST")

        self.assertEqual(len(records), len(self.mod.FAMILIES) * 2)
        self.assertEqual(summary["status"], "done")
        self.assertEqual(set(summary["family_counts"].values()), {2})

    def test_boundary_family_has_enough_near_edge_variants(self):
        records, summary = self.mod.generate_stress_split(per_family_count=61, run_id="R010", split_id="D5_TEST")

        self.assertEqual(summary["status"], "done")
        self.assertEqual(summary["family_counts"]["boundary_escape_route"], 61)

    def test_shared_liberty_count_detects_common_liberty(self):
        board = self.mod.TOPO.parse_board(
            [
                ".....",
                ".B.W.",
                "..B..",
                ".....",
                ".....",
            ]
        )

        self.assertGreaterEqual(self.mod.shared_liberty_count(board), 1)


if __name__ == "__main__":
    unittest.main()
