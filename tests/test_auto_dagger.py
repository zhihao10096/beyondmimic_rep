"""Controller tests with temporary fictional checkpoints/reports, never physical G2 evidence."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/stage2"))
spec = importlib.util.spec_from_file_location("auto_dagger_test_module", ROOT / "scripts/stage2/auto_dagger.py")
auto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto)
wf = auto.wf


class AutoDaggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(wf, "ROOT", self.root)
        self.root_patch.start()
        for name in ("auto_dagger.py", "workflow.py", "train.py", "rollout.py"):
            p = self.root / "scripts/stage2" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("UNIT TEST MOCK RUNTIME ONLY, NOT EXECUTABLE TRAINING CODE")
        self.rows = []
        for motion in ("fixture_a", "fixture_b"):
            gate_path = self.root / f"{motion}_fixture_gate.json"
            gate = {"teacher": {"fixture": motion}, "protocol": "full_motion_phase0_clean",
                    "collector_seed": 300, "num_envs": 20, "success_rate": 1.0, "body_error_mean_m": .02,
                    "fixture_notice": "UNIT TEST ONLY; not a valid actual teacher gate"}
            gate_path.write_text(json.dumps(gate))
            self.rows.append({"motion_id": motion, "admitted_for_initial_d0": True,
                              "motion_path": str(self.root/f"{motion}_NOT_REAL_MOTION.npz"),
                              "checkpoint_path": str(self.root/f"{motion}_NOT_REAL_TEACHER.pt"),
                              "agent_config_path": str(self.root/"NOT_REAL_AGENT.yaml"),
                              "env_config_path": str(self.root/"NOT_REAL_ENV.yaml"),
                              "mapping_hash": "UNIT_TEST_MAPPING", "motion_sha256": "UNIT_TEST_MOTION",
                              "checkpoint_sha256": "UNIT_TEST_TEACHER", "agent_sha256": "UNIT_TEST_AGENT",
                              "env_sha256": "UNIT_TEST_ENV", "gate_g1_report_path": str(gate_path),
                              "gate_g1_report_sha256": auto.file_hash(gate_path)})
        self.registry = self.root / "registry_fixture.json"
        self.registry.write_text(json.dumps({"contract_hash": auto.CONTRACT_HASH, "teachers": self.rows}))
        self.student = self.root / "logs/stage2/cvae_d1/best.pt"
        self.student.parent.mkdir(parents=True)
        self.student.write_text(json.dumps({"epoch": 0, "mapping_hash": "UNIT_TEST_MAPPING",
                                "normalizer_source": "UNIT_TEST_FROZEN_STATS", "train_fingerprint": "initial_fixture",
                                "fixture_notice": "NOT A TORCH CHECKPOINT; UNIT TEST ONLY"}))
        self.args = argparse.Namespace(student=str(self.student), start_round=2, registry=str(self.registry),
            data_root="data/stage2", model_root="logs/stage2", state_dir="artifacts/controller_fixture",
            device="cpu", num_envs=1, steps=2, val_steps=1, epochs=2, samples_per_epoch=4, batch_size=2,
            initial_summary=None, max_rounds=0, dry_run=False)
        self.calls = []
        self.pass_round = 3
        self.interrupt_collect = False
        self.interrupt_train = False
        for row in self.rows:
            for split in ("train", "val"):
                self.manifest(row, 0, split, self.student, 1)

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def info(self, p):
        result = json.loads(Path(p).read_text())
        return {**result, "sha256": auto.file_hash(p)}

    def manifest(self, row, round_id, split, student, steps, complete=True):
        directory = wf.run_directory(self.args.data_root, row["motion_id"], round_id, split)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"teacher_qualified": True, "motion_id": row["motion_id"], "round": round_id,
                "split": split, "collector_seed": wf.collector_seed(round_id, split), "num_envs": 1,
                "mapping_hash": row["mapping_hash"], "motion_sha256": row["motion_sha256"],
                "teacher": {"checkpoint_sha256": row["checkpoint_sha256"], "agent_sha256": row["agent_sha256"]},
                "env_yaml_sha256": row["env_sha256"], "teacher_gate_sha256": row["gate_g1_report_sha256"],
                "student_sha256": auto.file_hash(student), "executed_policy": "student_only"}
        manifest = {"complete": complete, "contract_hash": auto.CONTRACT_HASH, "metadata": meta,
                    "shards": [{"rows": steps, "path": "UNIT_TEST_NO_SHARD.npz"}],
                    "fixture_notice": "No tensors/shards exist. UNIT TEST ONLY, not valid actual training data"}
        (directory / "manifest.json").write_text(json.dumps(manifest))

    def evaluation(self, output, student, passed):
        output.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        for row in self.rows:
            gate = json.loads(Path(row["gate_g1_report_path"]).read_text())
            successes = 20 if passed else 0
            raw = {"student_sha256": auto.file_hash(student), "teacher": gate["teacher"],
                   "motion_sha256": row["motion_sha256"], "env_yaml_sha256": row["env_sha256"],
                   "mapping_hash": row["mapping_hash"], "contract_hash": auto.CONTRACT_HASH,
                   "implementation_sha256": {},
                   "teacher_gate_sha256": row["gate_g1_report_sha256"], "protocol": gate["protocol"],
                   "collector_seed": 300, "num_envs": 20, "evaluation_steps": 0, "policy": "student_only",
                   "executed_policy": "student_only", "latent": "mean", "baseline_comparable": True,
                   "success_definition": "reference_end_without_physical_failure", "trials": 20,
                   "successes": successes, "success_rate": successes / 20, "body_error_mean_m": .01,
                   "trial_success": [passed] * 20, "reference_completed": [passed] * 20,
                   "trial_ticks": [50 if passed else 1] * 20, "motion_frames": 50, "passed_g2_clean": passed,
                   "fixture_notice": "MOCK RESULTS ONLY; no physics has been run"}
            report_path = output.parent / f"{output.stem}_{row['motion_id']}_fiction.json"
            report_path.write_text(json.dumps(raw))
            entries.append({"motion_id": row["motion_id"], "report": str(report_path),
                            "report_sha256": auto.file_hash(report_path), "successes": successes,
                            "trials": 20, "body_error_mean_m": .01, "passed_g2_clean": passed})
        summary = {"student_sha256": auto.file_hash(student), "student": str(student), "perturbed": False,
                   "results": entries, "all_passed_g2_clean": passed, "fixture_notice": "UNIT TEST ONLY"}
        output.write_text(json.dumps(summary))
        return summary

    def child(self, command):
        self.calls.append(command)
        value = lambda flag: command[command.index(flag)+1]
        if "--mode" in command:
            row = next(r for r in self.rows if r["motion_id"] == Path(value("--output")).parent.name)
            self.manifest(row, int(value("--round")), value("--split"), Path(value("--student_checkpoint")),
                          int(value("--steps")), complete=not self.interrupt_collect)
            if self.interrupt_collect:
                self.interrupt_collect = False
                raise subprocess.CalledProcessError(99, command)
        elif command[3] == "evaluate":
            student = Path(value("--student"))
            round_id = int(student.parent.name.removeprefix("cvae_d"))
            self.evaluation(Path(value("--summary_output")), student, round_id >= self.pass_round)
        elif command[3] == "train-dagger":
            round_id = int(value("--round"))
            data = wf.datasets(self.rows, self.args.data_root, round_id, False)
            manifests = [json.loads((Path(p)/"manifest.json").read_text()) for p in data]
            info = {"epoch": 0 if self.interrupt_train else self.args.epochs-1, "mapping_hash": "UNIT_TEST_MAPPING",
                    "normalizer_source": "UNIT_TEST_FROZEN_STATS",
                    "train_fingerprint": auto.digest_json([m for m in manifests if m["metadata"]["split"] == "train"]),
                    "fixture_notice": "FICTIONAL CHECKPOINT, NOT A TORCH MODEL"}
            output = Path(value("--output"))
            output.mkdir(parents=True, exist_ok=True)
            for name in ("last.pt", "best.pt"):
                (output/name).write_text(json.dumps(info))
            if self.interrupt_train:
                self.interrupt_train = False
                raise subprocess.CalledProcessError(99, command)
        else:
            raise AssertionError(command)

    def run_mock(self):
        with patch.object(auto, "checkpoint_info", side_effect=self.info), \
             patch.object(wf, "verify_teacher"), patch.object(auto, "execute", side_effect=self.child):
            return auto.run(self.args)

    def test_loop_collects_all_motions_and_stops_only_after_clean_pass(self):
        self.assertEqual(self.run_mock(), 0)
        training = [c for c in self.calls if "train-dagger" in c]
        self.assertEqual(len(training), 2)
        state = json.loads((self.root/self.args.state_dir/"state.json").read_text())
        self.assertEqual(state["status"], "passed_g2_clean")
        self.assertEqual(state["next_round"], 4)
        self.assertTrue(state["current_student"].endswith("cvae_d3/best.pt"))
        self.assertEqual(sum("--mode" in c for c in self.calls), 8)

    def test_training_interrupt_resumes_last_without_recollecting(self):
        self.pass_round = 2
        self.interrupt_train = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_mock()
        collected = sum("--mode" in c for c in self.calls)
        self.assertEqual(self.run_mock(), 0)
        self.assertEqual(sum("--mode" in c for c in self.calls), collected)
        resumed = [c for c in self.calls if "train-dagger" in c][-1]
        self.assertIn("--resume", resumed)
        self.assertNotIn("--student", resumed)

    def test_existing_initial_evaluation_is_verified_and_reused(self):
        self.pass_round = 2
        old = self.root/"artifacts/stage2/student_evaluation/fixture_prior_eval/summary.json"
        self.evaluation(old, self.student, False)
        before = old.read_bytes()
        self.assertEqual(self.run_mock(), 0)
        self.assertEqual(sum("evaluate" in c for c in self.calls), 1)  # Only the new D2 checkpoint is evaluated.
        self.assertEqual(old.read_bytes(), before)

    def test_interrupted_collection_is_preserved_and_restarted(self):
        self.pass_round = 2
        self.interrupt_collect = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_mock()
        self.assertEqual(self.run_mock(), 0)
        backups = list((self.root/"data/stage2/fixture_a").glob("*.interrupted_*"))
        self.assertEqual(len(backups), 1)
        self.assertFalse(json.loads((backups[0]/"manifest.json").read_text())["complete"])

    def test_limit_is_not_a_pass_and_next_invocation_continues(self):
        self.args.max_rounds = 1
        self.assertEqual(self.run_mock(), 2)
        state = json.loads((self.root/self.args.state_dir/"state.json").read_text())
        self.assertEqual(state["status"], "limit_reached_not_passed")
        self.assertEqual(self.run_mock(), 0)

    def test_summary_requires_every_motion_and_consistent_flags(self):
        output = self.root/"scores_fixture.json"
        original = self.evaluation(output, self.student, False)
        self.assertFalse(auto.validate_summary(output, self.student, self.rows)[0])
        for mutate in (lambda d:d.update(all_passed_g2_clean=True),
                       lambda d:d["results"].pop(), lambda d:d.update(perturbed=True)):
            bad = copy.deepcopy(original)
            mutate(bad)
            output.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                auto.validate_summary(output, self.student, self.rows)

    def test_short_horizon_cannot_end_loop(self):
        output = self.root/"scores_fixture.json"
        summary = self.evaluation(output, self.student, True)
        raw_path = Path(summary["results"][0]["report"])
        raw = json.loads(raw_path.read_text())
        raw["evaluation_steps"] = 25
        raw_path.write_text(json.dumps(raw))
        summary["results"][0]["report_sha256"] = auto.file_hash(raw_path)
        output.write_text(json.dumps(summary))
        with self.assertRaises(ValueError):
            auto.validate_summary(output, self.student, self.rows)

    def test_collector_seeds_remain_disjoint_after_round99(self):
        train = {wf.collector_seed(r, "train") for r in range(2000)}
        val = {wf.collector_seed(r, "val") for r in range(2000)}
        self.assertEqual(len(train), 2000)
        self.assertEqual(len(val), 2000)
        self.assertFalse(train & val)
        self.assertEqual(wf.collector_seed(2, "train"), 102)
        self.assertEqual(wf.collector_seed(2, "val"), 202)


if __name__ == "__main__":
    unittest.main()
