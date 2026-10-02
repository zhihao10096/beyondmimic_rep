"""Resumable real DAgger loop. Stop only on verified clean G2, an error, or an explicit limit."""
import _bootstrap
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import workflow as wf
from beyondmimic_stage2.core import CONTRACT_HASH, atomic_json, digest_json, file_hash, load_student


def execute(command):
    print(shlex.join(command), flush=True)
    child = subprocess.Popen(command, cwd=wf.ROOT, start_new_session=True)
    try:
        code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        # Stop workflow and its Isaac/training grandchildren together, including Kit close hangs.
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)
        except ProcessLookupError:
            pass
        raise


def location(p):
    p = wf.path(p)
    return str(p.relative_to(wf.ROOT)) if p.is_relative_to(wf.ROOT) else str(p)


def checkpoint_info(checkpoint):
    _, saved = load_student(checkpoint, "cpu")
    if saved["allow_unqualified"]:
        raise ValueError("自动正式DAgger拒绝debug/unqualified student")
    return {"sha256": file_hash(checkpoint), "epoch": saved["epoch"], "mapping_hash": saved["mapping_hash"],
            "normalizer_source": saved["normalizer_d0_source"], "train_fingerprint": saved["train_fingerprint"]}


def validate_summary(summary_path, student, rows):
    """Bind every score to raw reports, this exact policy, and every frozen teacher gate."""
    summary_path = wf.path(summary_path)
    summary = json.loads(summary_path.read_text())
    student_sha = file_hash(student)
    if summary.get("perturbed") is not False or summary.get("student_sha256") != student_sha:
        raise ValueError("评估协议或student身份不匹配")
    entries = summary["results"]
    ids = [r["motion_id"] for r in entries]
    if len(ids) != len(set(ids)) or set(ids) != {r["motion_id"] for r in rows}:
        raise ValueError("评估必须覆盖冻结名单中的全部motion，不能缺项/重复")
    by_id = {r["motion_id"]: r for r in rows}
    implementation = {str(p.relative_to(wf.ROOT)): file_hash(p)
                      for p in (wf.ROOT/"source/whole_body_tracking/beyondmimic_stage2").glob("*.py")}
    passes = []
    for entry in entries:
        row = by_id[entry["motion_id"]]
        report_path = wf.path(entry["report"])
        if not report_path.is_file():
            report_path = summary_path.parent / Path(entry["report"]).name
        report_sha = file_hash(report_path)
        if entry.get("report_sha256", report_sha) != report_sha:
            raise ValueError("原始回放报告hash改变")
        report = json.loads(report_path.read_text())
        gate = json.loads(wf.path(row["gate_g1_report_path"]).read_text())
        if (report["student_sha256"] != student_sha or report["teacher"] != gate["teacher"]
                or report["motion_sha256"] != row["motion_sha256"]
                or report["env_yaml_sha256"] != row["env_sha256"]
                or report["mapping_hash"] != row["mapping_hash"]
                or report["contract_hash"] != CONTRACT_HASH
                or report.get("implementation_sha256") != implementation
                or report["teacher_gate_sha256"] != row["gate_g1_report_sha256"]
                or report["protocol"] != gate["protocol"]
                or report["protocol"] != "full_motion_phase0_clean"
                or report["collector_seed"] != gate["collector_seed"]
                or report["num_envs"] != gate["num_envs"]
                or report.get("evaluation_steps", 0) != 0
                or report["policy"] != "student_only" or report["latent"] != "mean"
                or report["executed_policy"] != "student_only"
                or report["success_definition"] != "reference_end_without_physical_failure"
                or report.get("baseline_comparable") is not True):
            raise ValueError(f"{row['motion_id']}: G2评估身份/协议不匹配")
        n, successes, error = report["trials"], report["successes"], report["body_error_mean_m"]
        if (n < 20 or n != report["num_envs"] or n != len(report["trial_success"]) or successes != sum(report["trial_success"])
                or not 0 <= successes <= n or not math.isfinite(error) or error < 0
                or report["success_rate"] != successes / n
                or report["reference_completed"] != report["trial_success"]
                or len(report["trial_ticks"]) != n
                or any(s and t < report["motion_frames"] for s, t in zip(report["trial_success"], report["trial_ticks"]))):
            raise ValueError("原始评估分母、计数或误差无效")
        passed = (successes / n >= .90 and gate["success_rate"] - successes / n <= .10 + 1e-8
                  and error <= 1.25 * gate["body_error_mean_m"])
        if (report.get("passed_g2_clean") is not passed or entry.get("passed_g2_clean") is not passed
                or entry["successes"] != successes or entry["trials"] != n
                or entry["body_error_mean_m"] != error):
            raise ValueError("G2标记与原始回放指标不一致")
        entry.update(report=str(report_path), report_sha256=report_sha)
        passes.append(passed)
        print(f"  {row['motion_id']}: {successes}/{n}, body={100*error:.2f}cm, G2={passed}", flush=True)
    passed = all(passes)
    if summary.get("all_passed_g2_clean") is not passed:
        raise ValueError("汇总G2标记与各motion不一致")
    return passed, summary


def existing_evaluation(student, rows):
    """Reuse only a complete, independently validated existing evaluation of these exact weights."""
    sha = file_hash(student)
    for p in sorted((wf.ROOT/"artifacts/stage2/student_evaluation").glob("*/summary.json"), reverse=True):
        try:
            candidate = json.loads(p.read_text())
            if candidate.get("student_sha256") != sha or candidate.get("perturbed") is not False:
                continue
            _, summary = validate_summary(p, student, rows)
            print(f"复用已有完整评估：{p}", flush=True)
            return summary
        except (ValueError, KeyError, TypeError, OSError):
            continue  # A partial/old incompatible external report is never used as a passing gate.
    return None


def run(args):
    if min(args.start_round, args.epochs, args.samples_per_epoch, args.batch_size,
           args.num_envs, args.steps, args.val_steps) <= 0 or args.max_rounds < 0:
        raise ValueError("轮次和训练/采集预算必须为正，max_rounds至少为0")
    registry_path, initial = wf.path(args.registry), wf.path(args.student)
    options = {k: v for k, v in vars(args).items() if k not in ("dry_run", "max_rounds", "state_dir", "initial_summary")}
    options["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    for key in ("registry", "student", "data_root", "model_root"):
        options[key] = location(options[key])
    if args.dry_run:
        common = ["--registry", str(registry_path), "--device", args.device, "--dry_run"]
        base = [sys.executable, "-u", str(wf.ROOT / "scripts/stage2/workflow.py")]
        for command in (base + ["evaluate", "--student", str(initial), *common],
                        base + ["collect-dagger", "--student", str(initial), "--round", str(args.start_round),
                                "--data_root", args.data_root, "--num_envs", str(args.num_envs), "--steps", str(args.steps),
                                "--val_steps", str(args.val_steps), *common],
                        base + ["train-dagger", "--student", str(initial), "--round", str(args.start_round),
                                "--data_root", args.data_root, "--output", str(wf.path(args.model_root)/f"cvae_d{args.start_round}"),
                                "--epochs", str(args.epochs), "--samples_per_epoch", str(args.samples_per_epoch),
                                "--batch_size", str(args.batch_size), *common]):
            execute(command)  # Children are dry-run only; never launch simulator/training.
        print("仅预览一次循环；没有写状态、采集数据、训练或评估。")
        return 0
    state_dir = wf.path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    with open(state_dir / "run.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("该状态目录已有自动DAgger进程运行")
        return locked_loop(args, state_dir, registry_path, initial, options)


def locked_loop(args, state_dir, registry_path, initial, options):
    state_path, frozen = state_dir/"state.json", state_dir/"teacher_registry.json"
    sources = {str(p.relative_to(wf.ROOT)): file_hash(p) for p in
               [*(wf.ROOT/"source/whole_body_tracking/beyondmimic_stage2").glob("*.py"),
                *(wf.ROOT/"scripts/stage2"/name for name in ("rollout.py", "train.py", "workflow.py", "auto_dagger.py"))]}
    initial_info = checkpoint_info(initial)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["options"] != options or state["initial_sha256"] != initial_info["sha256"] or state["sources"] != sources:
            raise ValueError("恢复参数、初始student或源码改变；不要混用实验，请指定新--state_dir")
        if file_hash(frozen) != state["registry_sha256"]:
            raise ValueError("冻结teacher名单改变")
    else:
        registry = json.loads(registry_path.read_text())
        wf.admitted(registry)
        atomic_json(frozen, registry)
        state = {"options": options, "initial_sha256": initial_info["sha256"], "sources": sources,
                 "registry_sha256": file_hash(frozen), "rounds": {}, "status": "running",
                 "current_student": location(initial), "current_sha256": initial_info["sha256"],
                 "next_round": args.start_round}
        atomic_json(state_path, state)
    registry = json.loads(frozen.read_text())
    rows = wf.admitted(registry)
    if registry["contract_hash"] != CONTRACT_HASH or {r["mapping_hash"] for r in rows} != {initial_info["mapping_hash"]}:
        raise ValueError("teacher库与student mapping/contract不同")
    for row in rows:
        wf.verify_teacher(row)
    base = [sys.executable, "-u", str(wf.ROOT/"scripts/stage2/workflow.py")]
    common = ["--registry", str(frozen), "--device", args.device]
    completed_this_run = 0
    while True:
        current = wf.path(state["current_student"])
        info = checkpoint_info(current)
        if info["sha256"] != state["current_sha256"]:
            raise ValueError("已登记student文件被改写")
        round_id = state["next_round"]
        evaluation = state_dir/"evaluations"/f"before_d{round_id}_{info['sha256'][:12]}.json"
        print(f"\n评估student={current}；下一轮D{round_id}", flush=True)
        if not evaluation.exists():
            if args.initial_summary and current == initial:
                _, summary = validate_summary(args.initial_summary, current, rows)
                atomic_json(evaluation, summary)
            else:
                existing = existing_evaluation(current, rows)
                if existing is not None:
                    atomic_json(evaluation, existing)
                else:
                    execute(base + ["evaluate", "--student", str(current), "--summary_output", str(evaluation), *common])
        passed, _ = validate_summary(evaluation, current, rows)
        state["evaluation_summary"] = location(evaluation)
        if passed:
            state["status"] = "passed_g2_clean"
            atomic_json(state_path, state)
            print(f"\n全部motion通过G2 clean。模型：{current}\n证据：{evaluation}\n状态：{state_path}", flush=True)
            return 0
        if args.max_rounds and completed_this_run >= args.max_rounds:
            state["status"] = "limit_reached_not_passed"
            atomic_json(state_path, state)
            print("达到本次训练轮数上限，G2尚未通过；原命令再次运行可继续。", flush=True)
            return 2
        record = state["rounds"].setdefault(str(round_id), {"parent_sha256": info["sha256"], "collection_started": [],
                                                         "training_started": False, "training_complete": False})
        if record["parent_sha256"] != info["sha256"]:
            raise ValueError("本轮采集parent student改变")
        state["status"] = "running"
        atomic_json(state_path, state)
        for row in rows:
            for split in ("train", "val"):
                directory = wf.run_directory(args.data_root, row["motion_id"], round_id, split)
                key = f"{row['motion_id']}/{split}"
                steps = args.steps if split == "train" else args.val_steps
                if directory.exists():
                    manifest_path = directory/"manifest.json"
                    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                    if not manifest.get("complete"):
                        if key not in record["collection_started"]:
                            raise ValueError(f"发现非本任务创建的未完成目录：{directory}")
                        backup = directory.with_name(directory.name + ".interrupted_" + wf.stamp())
                        directory.rename(backup)
                        print(f"中断数据保留在{backup}，重新采集该split。", flush=True)
                    else:
                        wf.verify_data(directory, row, round_id, split)
                        meta = manifest["metadata"]
                        if (meta.get("student_sha256") != info["sha256"] or meta.get("executed_policy") != "student_only"
                                or meta["num_envs"] != args.num_envs
                                or sum(s["rows"] for s in manifest["shards"]) != args.num_envs * steps
                                or meta.get("teacher_gate_sha256") != row["gate_g1_report_sha256"]):
                            raise ValueError(f"已有数据的student/预算/GATE不匹配：{directory}")
                        print(f"复用完整采集：{directory}", flush=True)
                        continue
                if key not in record["collection_started"]:
                    record["collection_started"].append(key)
                atomic_json(state_path, state)
                command = wf.rollout(row) + ["--mode", "collect", "--student_checkpoint", str(current),
                          "--round", str(round_id), "--split", split, "--seed", str(wf.collector_seed(round_id, split)),
                          "--num_envs", str(args.num_envs), "--steps", str(steps), "--output", str(directory),
                          "--device", args.device, "--headless", "--fast_exit"]
                execute(command)
                wf.verify_data(directory, row, round_id, split)
        inputs = wf.datasets(rows, args.data_root, round_id, False)
        manifests = [json.loads((Path(p)/"manifest.json").read_text()) for p in inputs]
        fingerprint = digest_json([m for m in manifests if m["metadata"]["split"] == "train"])
        output = wf.path(args.model_root)/f"cvae_d{round_id}"
        if output.exists() and not record["training_started"]:
            raise ValueError(f"训练输出已存在但不属于本任务：{output}；可选新的--model_root")
        record["training_started"] = True
        atomic_json(state_path, state)
        last = output/"last.pt"
        last_info = checkpoint_info(last) if last.exists() else None
        if last_info and (last_info["train_fingerprint"] != fingerprint
                          or last_info["normalizer_source"] != info["normalizer_source"]):
            raise ValueError("恢复checkpoint与聚合数据/冻结统计不匹配")
        if not last_info or last_info["epoch"] < args.epochs - 1:
            command = base + ["train-dagger", "--round", str(round_id), "--data_root", args.data_root,
                      "--output", str(output), "--epochs", str(args.epochs), "--samples_per_epoch", str(args.samples_per_epoch),
                      "--batch_size", str(args.batch_size), *common]
            command += ["--resume", str(last)] if last_info else ["--student", str(current)]
            execute(command)
        best = output/"best.pt"
        best_info = checkpoint_info(best)
        final_last = checkpoint_info(last)
        if (final_last["epoch"] < args.epochs - 1 or best_info["train_fingerprint"] != fingerprint
                or best_info["normalizer_source"] != info["normalizer_source"]):
            raise ValueError("训练未完成或best权重的聚合数据/统计不匹配")
        record.update(training_complete=True, best_student=location(best), best_sha256=best_info["sha256"])
        state.update(current_student=location(best), current_sha256=best_info["sha256"], next_round=round_id+1)
        atomic_json(state_path, state)
        completed_this_run += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", default="logs/stage2/cvae_d1/best.pt")
    p.add_argument("--start_round", type=int, default=2)
    p.add_argument("--registry", default=str(wf.REGISTRY))
    p.add_argument("--data_root", default="data/stage2")
    p.add_argument("--model_root", default="logs/stage2")
    p.add_argument("--state_dir", default="artifacts/stage2/auto_dagger")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--val_steps", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--samples_per_epoch", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--initial_summary", help="Existing full clean evaluation of this exact initial student, with raw reports")
    p.add_argument("--max_rounds", type=int, default=0, help="Training rounds per invocation; 0=until pass/error/manual stop")
    p.add_argument("--dry_run", action="store_true")
    return run(p.parse_args())


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n已中断；原命令重跑可继续。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"自动DAgger停止（未宣称通过）：{error}", file=sys.stderr)
        sys.exit(1)
