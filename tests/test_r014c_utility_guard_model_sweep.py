import importlib.util
import os
import pathlib
import unittest

import numpy as np


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "scripts" / "r014c_utility_guard_model_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("r014c_utility_guard_model_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class R014CUtilityGuardModelSweepTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()

    def test_build_sweep_configs_has_unique_names_and_b3_gbdt_baseline(self):
        configs = self.mod.build_sweep_configs()
        names = [config["name"] for config in configs]

        self.assertEqual(len(names), len(set(names)))
        self.assertIn("S01_b3_gbdt_p1", names)

    def test_make_model_supports_expected_probability_estimators(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
        y = np.array([0, 0, 1, 1], dtype=int)

        for kind in ["logreg", "gbdt", "rf", "extra"]:
            model = self.mod.make_model(kind, random_state=0)
            model.fit(x, y)
            probabilities = model.predict_proba(x)
            self.assertEqual(probabilities.shape, (4, 2))

    def test_tree_model_parallelism_is_env_controlled(self):
        old_value = os.environ.get("GOTOPOGUARD_MODEL_N_JOBS")
        try:
            os.environ.pop("GOTOPOGUARD_MODEL_N_JOBS", None)
            self.assertEqual(self.mod.make_model("extra", random_state=0).n_jobs, 1)
            os.environ["GOTOPOGUARD_MODEL_N_JOBS"] = "8"
            self.assertEqual(self.mod.make_model("extra", random_state=0).n_jobs, 8)
            self.assertEqual(self.mod.make_model("rf", random_state=0).n_jobs, 8)
        finally:
            if old_value is None:
                os.environ.pop("GOTOPOGUARD_MODEL_N_JOBS", None)
            else:
                os.environ["GOTOPOGUARD_MODEL_N_JOBS"] = old_value

    def test_score_name_is_stable(self):
        config = {
            "feature_profile": "B3",
            "model_kind": "gbdt",
            "penalty": 1.5,
            "max_trigger_rate": 0.75,
        }

        self.assertEqual(self.mod.config_name("S99", config), "S99_b3_gbdt_p1p5_cap75")

    def test_stable_seed_is_reproducible(self):
        self.assertEqual(
            self.mod.stable_seed("S09_b3_extra_p1p5", "D5", "cyclic_enclosure"),
            self.mod.stable_seed("S09_b3_extra_p1p5", "D5", "cyclic_enclosure"),
        )
        self.assertNotEqual(
            self.mod.stable_seed("S09_b3_extra_p1p5", "D5", "cyclic_enclosure"),
            self.mod.stable_seed("S09_b3_extra_p1p5", "D5", "ladder_net_escape"),
        )


if __name__ == "__main__":
    unittest.main()
