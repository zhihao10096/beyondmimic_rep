"""Test diagnostic-horizon grading with a mock backend, never publish real gates."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
sys.path.insert(0, str(ROOT / "scripts/stage2"))


class LauncherStub:
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cpu")


class TeacherStub:
    metadata = {"fixture": "unit_test_only_no_checkpoint_weights"}

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, obs):
        return torch.zeros(len(obs), 29)


class AdapterStub:
    def __init__(self, count):
        self.n, self.tick = count, 0
        self.mapping, self.mapping_hash = {}, "unit_test_mapping"
        self.command = SimpleNamespace(motion=SimpleNamespace(time_step_total=500),
                                       time_steps=torch.zeros(count, dtype=torch.long),
                                       finished=torch.zeros(count, dtype=torch.bool))

    def snapshot(self):
        return {"reference": torch.zeros(self.n, 67), "proprio": torch.zeros(self.n, 96),
                "teacher_observation": torch.zeros(self.n, 160)}

    def body_error(self, row):
        return torch.zeros(self.n)

    def transition(self, action):
        self.tick += 1
        self.command.finished[:] = self.tick >= 500
        return torch.zeros(self.n), torch.zeros(self.n, dtype=torch.bool), torch.zeros(self.n, dtype=torch.bool), {
            "reasons": torch.zeros(self.n, 1, dtype=torch.bool)}


class TeacherEvaluationTests(unittest.TestCase):
    def run_fixture(self, horizon):
        fake_isaaclab = ModuleType("isaaclab")
        fake_app = ModuleType("isaaclab.app")
        fake_app.AppLauncher = LauncherStub
        fake_backend = ModuleType("beyondmimic_stage2.isaac")
        fake_backend.TASK = "UNIT_TEST_ONLY_NOT_A_REAL_ROBOT_ENVIRONMENT"

        def create_environment(motion, num_envs, *args, **kwargs):
            env = SimpleNamespace(step_dt=.04, scene=SimpleNamespace(env_origins=torch.zeros(num_envs, 3)),
                                  termination_manager=SimpleNamespace(active_terms=["unit_failure"]))
            return env, AdapterStub(num_envs)

        fake_backend.create_environment = create_environment
        with patch.dict(sys.modules, {"isaaclab": fake_isaaclab, "isaaclab.app": fake_app,
                                      "beyondmimic_stage2.isaac": fake_backend}), tempfile.TemporaryDirectory() as tmp:
            spec = importlib.util.spec_from_file_location("unit_rollout", ROOT / "scripts/stage2/rollout.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fixture = Path(tmp) / "fixture_not_real_motion.bin"
            fixture.write_bytes(b"UNIT TEST FIXTURE ONLY")
            output = Path(tmp) / "unit_report.json"
            p = module.parser()
            args = p.parse_args(["--mode", "evaluate", "--motion_file", str(fixture), "--teacher_checkpoint", str(fixture),
                                 "--agent_yaml", str(fixture), "--env_yaml", str(fixture), "--output", str(output),
                                 "--num_envs", "20", "--evaluation_steps", str(horizon), "--device", "cpu"])
            module.preflight(p, args)
            with patch("beyondmimic_stage2.teacher.Teacher", TeacherStub):
                module.main(args, None)
            return json.loads(output.read_text())

    def test_short_horizon_survival_never_qualifies_g1(self):
        result = self.run_fixture(2)
        self.assertEqual(result["successes"], 20)
        self.assertFalse(result["passed_g1"])
        self.assertTrue(all(result["survived_horizon"]))
        self.assertFalse(any(result["reference_completed"]))
        self.assertEqual(result["trial_ticks"], [2] * 20)

    def test_full_reference_end_is_distinct_from_horizon(self):
        result = self.run_fixture(0)
        self.assertEqual(result["successes"], 20)
        self.assertTrue(result["passed_g1"])
        self.assertFalse(any(result["survived_horizon"]))
        self.assertTrue(all(result["reference_completed"]))
        self.assertEqual(result["trial_ticks"], [500] * 20)


if __name__ == "__main__":
    unittest.main()
