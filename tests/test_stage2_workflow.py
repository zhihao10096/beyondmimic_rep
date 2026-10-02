"""Workflow guard tests. Temporary manifest fixtures are never physical rollout data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/stage2"))
spec = importlib.util.spec_from_file_location("stage2_workflow", ROOT / "scripts/stage2/workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def row(motion):
    return {"motion_id": motion, "admitted_for_initial_d0": True, "mapping_hash": "UNIT_TEST_ONLY",
            "motion_sha256": "fixture_motion", "checkpoint_sha256": "fixture_checkpoint",
            "agent_sha256": "fixture_agent", "env_sha256": "fixture_environment"}


def fixture(directory, teacher, round_id, split):
    directory.mkdir(parents=True)
    metadata = {"teacher_qualified": True, "motion_id": teacher["motion_id"], "round": round_id,
                "split": split, "collector_seed": (100 if split == "train" else 200) + round_id,
                "mapping_hash": teacher["mapping_hash"], "motion_sha256": teacher["motion_sha256"],
                "teacher": {"checkpoint_sha256": teacher["checkpoint_sha256"], "agent_sha256": teacher["agent_sha256"]},
                "env_yaml_sha256": teacher["env_sha256"]}
    manifest = {"complete": True, "contract_hash": workflow.CONTRACT_HASH, "metadata": metadata,
                "fixture_notice": "UNIT TEST ONLY; no shards, not valid training or physical gate data"}
    (directory / "manifest.json").write_text(json.dumps(manifest))


class WorkflowGuardTests(unittest.TestCase):
    def test_training_machine_needs_verified_manifests_but_not_teacher_weights(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(workflow, "ROOT", Path(tmp)):
            teacher = row("fixture")
            registry = Path(tmp) / "registry.json"
            registry.write_text(json.dumps({"contract_hash": workflow.CONTRACT_HASH, "teachers": [teacher]}))
            for split in ("train", "val"):
                fixture(workflow.run_directory("data/stage2", teacher["motion_id"], 0, split), teacher, 0, split)
            with patch.object(sys, "argv", ["workflow.py", "train-d0", "--registry", str(registry)]), \
                 patch.object(workflow, "verify_teacher", side_effect=AssertionError("No teacher weights on this machine")), \
                 patch.object(workflow, "execute") as execute:
                workflow.main()
            execute.assert_called_once()
            self.assertIn("--data", execute.call_args.args[0])
            self.assertFalse(execute.call_args.args[1])

    def test_failed_whole_motion_or_failed_robustness_cannot_be_admitted(self):
        for clean_pass, robust_rate in ((False, 1.0), (True, .90)):
            candidate = {"full_clean": {"status": "complete", "passed_g1": clean_pass, "trials": 20, "success_rate": 1.0},
                         "random_perturbed_10s": {"status": "complete", "trials": 20, "success_rate": robust_rate}}
            with self.assertRaisesRegex(ValueError, "registry unchanged"):
                workflow.candidate_entry(candidate)

    def test_expanded_motion_requires_d0_and_current_round_but_not_old_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            old, new = row("old_fixture"), row("new_fixture")
            for teacher, rounds in ((old, (0, 1, 2)), (new, (0, 2))):
                for r in rounds:
                    for split in ("train", "val"):
                        fixture(workflow.run_directory(data, teacher["motion_id"], r, split), teacher, r, split)
            inputs = workflow.datasets([old, new], data, 2, False)
            self.assertEqual(len(inputs), 10)
            self.assertTrue(any("old_fixture/d1_train_s101" in p for p in inputs))
            self.assertFalse(any("new_fixture/d1_" in p for p in inputs))
            (workflow.run_directory(data, new["motion_id"], 2, "val") / "manifest.json").unlink()
            with self.assertRaises(FileNotFoundError):
                workflow.datasets([old, new], data, 2, False)

    def test_replaced_teacher_rejects_old_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = row("fixture")
            directory = Path(tmp) / "fixture"
            fixture(directory, teacher, 0, "train")
            replacement = {**teacher, "checkpoint_sha256": "a_different_fixture_checkpoint"}
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                workflow.verify_data(directory, replacement, 0, "train")

    def test_registry_update_merges_with_latest_disk_state(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(workflow, "ROOT", Path(tmp)):
            p = Path(tmp) / "registry_fixture.json"
            # Another writer's addition already exists before this writer commits.
            p.write_text(json.dumps({"contract_hash": workflow.CONTRACT_HASH,
                                     "teachers": [row("original"), row("other_writer")]}))
            workflow.register_candidate(p, row("this_writer"), Path(tmp) / "fixture_summary_not_real.json")
            saved = json.loads(p.read_text())
            self.assertEqual(set(saved["admitted_motion_ids"]), {"original", "other_writer", "this_writer"})
            self.assertEqual(saved["teachers"][-1]["evaluation_summary_path"], "fixture_summary_not_real.json")


if __name__ == "__main__":
    unittest.main()
