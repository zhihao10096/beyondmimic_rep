"""Short stage2 commands backed by the evaluated teacher registry; no Isaac import."""
import _bootstrap
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

from beyondmimic_stage2.core import CONTRACT_HASH, atomic_json, file_hash

ROOT = _bootstrap.ROOT
REGISTRY = ROOT / "docs/beyondmimic_reproduction/configs/teacher_registry_current.json"


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def relative(value):
    return str(path(value).relative_to(ROOT))


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")


def admitted(registry, motion="all"):
    rows = [r for r in registry["teachers"] if r["admitted_for_initial_d0"]]
    ids = [r["motion_id"] for r in registry["teachers"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Teacher registry has duplicate motion IDs")
    if motion != "all":
        rows = [r for r in rows if r["motion_id"] == motion]
    if not rows:
        raise ValueError(f"No admitted teacher for {motion}; use add-teacher after preparing its checkpoint")
    return rows


def verify_teacher(row):
    for key, hash_key in (("motion_path", "motion_sha256"), ("checkpoint_path", "checkpoint_sha256"),
                          ("agent_config_path", "agent_sha256"), ("env_config_path", "env_sha256"),
                          ("gate_g1_report_path", "gate_g1_report_sha256"),
                          ("robustness_report_path", "robustness_report_sha256")):
        if file_hash(path(row[key])) != row[hash_key]:
            raise ValueError(f"{row['motion_id']}: {key} changed; re-evaluate this candidate")
    gate = json.loads(path(row["gate_g1_report_path"]).read_text())
    robust = json.loads(path(row["robustness_report_path"]).read_text())
    if not (gate["passed_g1"] and gate["trials"] >= 20 and gate["success_rate"] >= .95
            and robust["trials"] >= 20 and robust["success_rate"] >= .95):
        raise ValueError(f"{row['motion_id']}: teacher does not meet admission criteria")
    for report in (gate, robust):
        if (report["teacher"]["checkpoint_sha256"] != row["checkpoint_sha256"]
                or report["teacher"]["agent_sha256"] != row["agent_sha256"]
                or report["motion_sha256"] != row["motion_sha256"]
                or report["env_yaml_sha256"] != row["env_sha256"]
                or report["mapping_hash"] != row["mapping_hash"]
                or report["contract_hash"] != CONTRACT_HASH):
            raise ValueError(f"{row['motion_id']}: report identity mismatch")


def rollout(row):
    return [sys.executable, "-u", str(ROOT / "scripts/stage2/rollout.py"),
            "--motion_file", str(path(row["motion_path"])),
            "--teacher_checkpoint", str(path(row["checkpoint_path"])),
            "--agent_yaml", str(path(row["agent_config_path"])),
            "--env_yaml", str(path(row["env_config_path"])),
            "--teacher_gate", str(path(row["gate_g1_report_path"]))]


def execute(command, dry_run):
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_directory(data_root, motion, round_id, split):
    seed = collector_seed(round_id, split)
    return path(data_root) / motion / f"d{round_id}_{split}_s{seed}"


def collector_seed(round_id, split):
    if round_id < 0 or split not in ("train", "val"):
        raise ValueError("Invalid collector round or split")
    # Keep existing d0..d99 paths; unbounded automation must never cross train/val seeds.
    if round_id < 100:
        return (100 if split == "train" else 200) + round_id
    return 1000 + 2 * round_id + (split == "val")


def verify_data(directory, row, round_id, split):
    manifest = json.loads((directory / "manifest.json").read_text())
    meta = manifest["metadata"]
    if not manifest.get("complete") or not meta.get("teacher_qualified"):
        raise ValueError(f"Incomplete or debug data: {directory}")
    if (manifest["contract_hash"] != CONTRACT_HASH or meta["motion_id"] != row["motion_id"]
            or meta["round"] != round_id or meta["split"] != split
            or meta["collector_seed"] != collector_seed(round_id, split)
            or meta["mapping_hash"] != row["mapping_hash"]
            or meta["motion_sha256"] != row["motion_sha256"]
            or meta["teacher"]["checkpoint_sha256"] != row["checkpoint_sha256"]
            or meta["teacher"]["agent_sha256"] != row["agent_sha256"]
            or meta["env_yaml_sha256"] != row["env_sha256"]):
        raise ValueError(f"Data no longer matches the registered teacher/partition: {directory}; use fresh collection directories")


def datasets(rows, data_root, last_round, dry_run):
    result = []
    for row in rows:
        for round_id in range(last_round + 1):
            for split in ("train", "val"):
                directory = run_directory(data_root, row["motion_id"], round_id, split)
                # New motions need D0 and the current round, but never fake old DAgger rounds.
                required = round_id in (0, last_round)
                if not required and not directory.exists():
                    continue
                if not dry_run:
                    verify_data(directory, row, round_id, split)
                result.append(str(directory))
    return result


def candidate_entry(candidate):
    clean, robust = candidate["full_clean"], candidate["random_perturbed_10s"]
    if not (clean["status"] == robust["status"] == "complete" and clean["passed_g1"]
            and clean["trials"] >= 20 and clean["success_rate"] >= .95
            and robust["trials"] >= 20 and robust["success_rate"] >= .95):
        raise ValueError("Candidate failed admission; registry unchanged. Inspect the saved evaluation summary")
    report = json.loads(path(clean["report"]).read_text())
    return {"motion_id": candidate["motion_id"], "admitted_for_initial_d0": True,
            "status": "recommended", "recommendation": "推荐进入正式teacher库",
            "iteration": candidate["iteration"], "motion_path": relative(candidate["motion_file"]),
            "motion_sha256": candidate["motion_sha256"], "checkpoint_path": relative(candidate["checkpoint"]),
            "checkpoint_sha256": candidate["teacher"]["checkpoint_sha256"],
            "agent_config_path": relative(candidate["agent_yaml"]), "agent_sha256": candidate["teacher"]["agent_sha256"],
            "env_config_path": relative(candidate["env_yaml"]), "env_sha256": candidate["env_yaml_sha256"],
            "normalizer_source": "same_teacher_checkpoint", "mapping_hash": report["mapping_hash"],
            "gate_g1_report_path": relative(clean["report"]), "gate_g1_report_sha256": clean["report_sha256"],
            "gate_g1_passed": True, "full_clean_successes": clean["successes"], "full_clean_trials": clean["trials"],
            "robustness_report_path": relative(robust["report"]), "robustness_report_sha256": robust["report_sha256"],
            "robustness_successes": robust["successes"], "robustness_trials": robust["trials"]}


def register_candidate(registry_path, row, summary_path):
    # Parallel candidate evaluations must merge into the latest registry, not a stale snapshot.
    import fcntl
    import hashlib
    lock_dir = ROOT / "artifacts/stage2/registry_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / (hashlib.sha256(str(registry_path).encode()).hexdigest() + ".lock")
    with open(lock, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        registry = json.loads(registry_path.read_text())
        if registry["contract_hash"] != CONTRACT_HASH:
            raise ValueError("Registry contract changed during evaluation")
        for current in admitted(registry):
            if current["mapping_hash"] != row["mapping_hash"]:
                raise ValueError("New teacher asset/action mapping differs from the existing library")
        row["evaluation_summary_path"] = relative(summary_path)
        registry["teachers"] = [r for r in registry["teachers"] if r["motion_id"] != row["motion_id"]] + [row]
        registry["admitted_motion_ids"] = [r["motion_id"] for r in admitted(registry)]
        registry["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        atomic_json(registry_path, registry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("collect-d0", "train-d0", "evaluate", "collect-dagger", "train-dagger", "add-teacher"):
        p = sub.add_parser(name)
        p.add_argument("--registry", default=str(REGISTRY))
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--dry_run", action="store_true", help="Print commands only; never launch or modify files")
        if name in ("collect-d0", "collect-dagger", "evaluate"):
            p.add_argument("--motion", default="all")
            p.add_argument("--num_envs", type=int, default=20 if name == "evaluate" else 32)
        if name in ("collect-d0", "collect-dagger", "train-d0", "train-dagger"):
            p.add_argument("--data_root", default="data/stage2")
        if name in ("collect-d0", "collect-dagger"):
            p.add_argument("--steps", type=int, default=2500)
            p.add_argument("--val_steps", type=int, default=1000)
        if name in ("collect-dagger", "evaluate"):
            p.add_argument("--student", required=True)
        if name in ("collect-dagger", "train-dagger"):
            p.add_argument("--round", type=int, required=True)
        if name in ("train-d0", "train-dagger"):
            p.add_argument("--output")
            p.add_argument("--resume", help="Exact interrupted-run resume")
            for option in ("epochs", "samples_per_epoch", "batch_size"):
                p.add_argument(f"--{option}", type=int)
        if name == "train-dagger":
            p.add_argument("--student", help="Previous CVAE checkpoint; starts a new dataset round")
        if name == "evaluate":
            p.add_argument("--perturbed", action="store_true", help="Full-motion perturbation diagnostic, separate from clean G2")
            p.add_argument("--summary_output", help="Also write summary to this fresh path for automation")
        if name == "add-teacher":
            p.add_argument("--motion", required=True)
            p.add_argument("--teacher_root", default="logs/rsl_rl/g1_flat")
    args = parser.parse_args()
    if getattr(args, "round", 1) < 1 or getattr(args, "num_envs", 1) < 1:
        parser.error("round and num_envs must be positive")
    if min(getattr(args, "steps", 1), getattr(args, "val_steps", 1)) <= 0:
        parser.error("steps and val_steps must be positive")
    registry_path = path(args.registry)
    registry = json.loads(registry_path.read_text())
    registry_sha256 = file_hash(registry_path)
    if registry["contract_hash"] != CONTRACT_HASH:
        raise ValueError("Registry contract mismatch")
    if args.action == "add-teacher":
        output = ROOT / "artifacts/stage2/teacher_candidates" / f"{args.motion}_{stamp()}"
        command = [sys.executable, "-u", str(ROOT / "scripts/stage2/evaluate_teachers.py"),
                   "--root", str(path(args.teacher_root)), "--motions", args.motion,
                   "--output", str(output), "--num_envs", "20", "--device", args.device]
        if args.motion == "all" or Path(args.motion).name != args.motion:
            parser.error("Specify one motion ID, not all or a path")
        # Refuse ambiguity before running a potentially long batch.
        import yaml
        matches = []
        for env_yaml in path(args.teacher_root).glob("*/params/env.yaml"):
            saved = yaml.load(env_yaml.read_text(), Loader=yaml.BaseLoader)
            if Path(saved["commands"]["motion"]["motion_file"]).stem == args.motion and list(env_yaml.parent.parent.glob("model_*.pt")):
                matches.append(env_yaml)
        if len(matches) != 1:
            raise ValueError(f"Expected one checkpoint run for {args.motion}, found {len(matches)} in {args.teacher_root}")
        execute(command, args.dry_run)
        if args.dry_run:
            return
        summary = json.loads((output / "summary.json").read_text())
        if not summary["complete"] or len(summary["candidates"]) != 1:
            raise ValueError("Candidate evaluation incomplete or ambiguous; registry unchanged")
        row = candidate_entry(summary["candidates"][0])
        verify_teacher(row)
        register_candidate(registry_path, row, output / "summary.json")
        print(f"Admitted {args.motion}: {registry_path}; reports: {output}")
        return
    rows = admitted(registry, getattr(args, "motion", "all"))
    # Training can run on a separate machine containing only the registry and collected data.
    # Its manifests are checked against each registered identity below, then fully checked by train.py.
    if args.action not in ("train-d0", "train-dagger"):
        for row in rows:
            verify_teacher(row)
    if getattr(args, "student", None) and not args.dry_run and not path(args.student).is_file():
        raise FileNotFoundError(args.student)
    if args.action in ("collect-d0", "collect-dagger"):
        round_id = 0 if args.action == "collect-d0" else args.round
        commands = []
        for row in rows:
            for split in ("train", "val"):
                output = run_directory(args.data_root, row["motion_id"], round_id, split)
                if output.exists() and not args.dry_run:
                    raise ValueError(f"Output exists: {output}; choose a new --data_root or round")
                command = rollout(row) + ["--mode", "collect", "--round", str(round_id), "--split", split,
                           "--seed", str(collector_seed(round_id, split)),
                           "--num_envs", str(args.num_envs), "--steps", str(args.steps if split == "train" else args.val_steps),
                           "--output", str(output), "--device", args.device, "--headless", "--fast_exit"]
                if round_id:
                    command += ["--student_checkpoint", str(path(args.student))]
                commands.append(command)
        for command in commands:
            execute(command, args.dry_run)
    elif args.action in ("train-d0", "train-dagger"):
        round_id = 0 if args.action == "train-d0" else args.round
        if round_id and bool(args.student) == bool(args.resume):
            parser.error("train-dagger needs either --student for a new round or --resume for interrupted-run recovery")
        data = datasets(rows, args.data_root, round_id, args.dry_run)
        output = path(args.output or f"logs/stage2/cvae_d{round_id}")
        command = [sys.executable, "-u", str(ROOT / "scripts/stage2/train.py"), "--data", *data,
                   "--output", str(output), "--device", args.device]
        if args.resume:
            command += ["--resume", str(path(args.resume))]
        elif round_id:
            command += ["--resume", str(path(args.student)), "--new_round"]
        for option in ("epochs", "samples_per_epoch", "batch_size"):
            if getattr(args, option) is not None:
                command += [f"--{option}", str(getattr(args, option))]
        execute(command, args.dry_run)
    elif args.action == "evaluate":
        if args.summary_output and path(args.summary_output).exists() and not args.dry_run:
            raise ValueError(f"Evaluation summary output exists: {args.summary_output}")
        output = ROOT / "artifacts/stage2/student_evaluation" / f"{path(args.student).parent.name}_{stamp()}"
        results = []
        for row in rows:
            report = output / f"{row['motion_id']}.json"
            command = rollout(row) + ["--mode", "evaluate", "--student_checkpoint", str(path(args.student)),
                       "--num_envs", str(args.num_envs), "--seed", "400" if args.perturbed else "300",
                       "--output", str(report), "--device", args.device, "--headless", "--fast_exit"]
            if args.perturbed:
                command += ["--perturbed"]
            execute(command, args.dry_run)
            if not args.dry_run:
                r = json.loads(report.read_text())
                results.append({"motion_id": row["motion_id"], "report": str(report), "successes": r["successes"],
                                "report_sha256": file_hash(report),
                                "trials": r["trials"], "body_error_mean_m": r["body_error_mean_m"],
                                "passed_g2_clean": r.get("passed_g2_clean")})
        if not args.dry_run:
            summary = {"student": str(path(args.student)), "student_sha256": file_hash(path(args.student)),
                        "registry_sha256": registry_sha256,
                        "perturbed": args.perturbed, "results": results,
                        "all_passed_g2_clean": None if args.perturbed else all(r["passed_g2_clean"] is True for r in results)}
            atomic_json(output / "summary.json", summary)
            if args.summary_output:
                atomic_json(path(args.summary_output), summary)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            print(f"Student evaluation summary: {output / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f"Stage2 workflow stopped: {error}", file=sys.stderr)
        sys.exit(1)
