"""Audit candidate checkpoints and run sequential real-physics teacher evaluations."""
import _bootstrap
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from beyondmimic_stage2.core import CONTRACT_HASH, atomic_json, file_hash, runtime_versions
from beyondmimic_stage2.teacher import Teacher


def verdict(row):
    clean = row.get("full_clean", {})
    robust = row.get("random_perturbed_10s", {})
    if clean.get("status") == "error":
        return "接口/执行失败，未完成质量评估"
    if clean.get("status") != "complete":
        return "评估中"
    if not clean.get("passed_g1"):
        if robust.get("status") == "complete" and robust["success_rate"] >= .95:
            return "完整动作未合格；可筛选片段后重新验收"
        return "当前不推荐进入正式teacher库"
    if robust.get("status") != "complete":
        return "G1 clean合格；扰动评估待完成"
    if robust["success_rate"] >= .95:
        return "推荐进入正式teacher库"
    return "G1 clean合格；随机phase扰动不足，先小批D0验证"


def summarize(output, manifest):
    for row in manifest["candidates"]:
        row["recommendation"] = verdict(row)
    atomic_json(output / "summary.json", manifest)
    lines = ["# Teacher checkpoint评估", "", f"日期：{manifest['started_at']}。本机实际物理回放，未启动teacher训练。", "",
             "G1：phase0到动作末尾20次，clean保留startup随机化，成功率≥95%。补充：随机phase、原始reset噪声及interval push，20次10秒，观测保持clean以匹配stage2标签接口。两个分母分别统计。", "",
             "| Motion | Iteration | 完整clean成功 | 平均body误差(cm) | 随机扰动10s成功 | 建议 |", "|---|---:|---:|---:|---:|---|"]
    for row in manifest["candidates"]:
        a, b = row.get("full_clean", {}), row.get("random_perturbed_10s", {})
        clean = f"{a['successes']}/{a['trials']}" if a.get("status") == "complete" else a.get("status", "pending")
        robust = f"{b['successes']}/{b['trials']}" if b.get("status") == "complete" else b.get("status", "pending")
        error = f"{100*a['body_error_mean_m']:.2f}" if a.get("status") == "complete" else "—"
        lines.append(f"| {row['motion_id']} | {row['iteration']} | {clean} | {error} | {robust} | {row['recommendation']} |")
    lines += ["", "完整motion门槛不能用10秒存活替代。未合格motion若要切片入库，必须对切片参考和对应teacher重新验收；不自动删除失败帧或降低门槛。", "",
              "每个motion的原始JSON、终止原因、trial长度、初始phase、checkpoint/asset/配置hash与运行日志保存在本目录。20个并行trial使用同一collector seed的独立随机化抽样，不是20个独立训练seed。", ""]
    (output / "summary.md").write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="logs/rsl_rl/g1_flat")
    p.add_argument("--motions", nargs="+", help="Optional motion IDs")
    p.add_argument("--motion_root", default="data/reproduction/motions_25hz")
    p.add_argument("--output", required=True)
    p.add_argument("--num_envs", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clean_seed", type=int, default=300)
    p.add_argument("--robust_seed", type=int, default=400)
    p.add_argument("--job_timeout", type=int, default=1800)
    p.add_argument("--inventory_only", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip only reports verified against immutable run hashes")
    args = p.parse_args()
    if args.num_envs < 20 or args.job_timeout <= 0:
        p.error("At least20 trials and positive job_timeout required")
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        p.error("Output exists; select fresh output or explicit --resume")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(Path(args.root).glob("*")):
        checkpoints = list(run.glob("model_*.pt"))
        if not checkpoints:
            continue
        checkpoint = max(checkpoints, key=lambda f: int(f.stem.split("_")[1]))
        env_yaml, agent_yaml = run / "params/env.yaml", run / "params/agent.yaml"
        cfg = yaml.load(env_yaml.read_text(), Loader=yaml.BaseLoader)
        motion = Path(args.motion_root) / Path(cfg["commands"]["motion"]["motion_file"]).name
        if args.motions and motion.stem not in args.motions:
            continue
        teacher = Teacher(checkpoint, agent_yaml)
        with torch.no_grad():
            if not torch.isfinite(teacher(torch.zeros(1, 160))).all():
                raise ValueError(f"Nonfinite actor output: {checkpoint}")
        with np.load(motion) as data:
            frames, fps = len(data["joint_pos"]), float(data["fps"].reshape(-1)[0])
        if fps != 25 or float(cfg["sim"]["dt"]) * int(cfg["decimation"]) != .04:
            raise ValueError(f"Candidate does not use25Hz: {run}")
        rows.append({"motion_id": motion.stem, "motion_file": str(motion.resolve()),
                     "motion_sha256": file_hash(motion), "motion_frames": frames, "duration_s": frames/fps,
                     "checkpoint": str(checkpoint.resolve()), "iteration": int(checkpoint.stem.split("_")[1]),
                     "agent_yaml": str(agent_yaml.resolve()), "env_yaml": str(env_yaml.resolve()),
                     "env_yaml_sha256": file_hash(env_yaml), "teacher": teacher.metadata,
                     "full_clean": {"status": "pending"}, "random_perturbed_10s": {"status": "pending"}})
    if not rows:
        p.error("No matching checkpoints")
    implementation = {str(f.relative_to(_bootstrap.ROOT)): file_hash(f) for f in
                      [*(_bootstrap.ROOT / "source/whole_body_tracking/beyondmimic_stage2").glob("*.py"),
                       _bootstrap.ROOT / "scripts/stage2/rollout.py", Path(__file__).resolve()]}
    manifest = {"started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "contract_hash": CONTRACT_HASH,
                "runtime_versions": runtime_versions(), "source_sha256": implementation,
                "profiles": {"full_clean": {"seed": args.clean_seed, "trials": args.num_envs},
                             "random_perturbed_10s": {"seed": args.robust_seed, "trials": args.num_envs, "horizon_ticks": 250}},
                "candidates": rows, "complete": False}
    summarize(output, manifest)
    print(f"Audited{len(rows)} candidates; reports: {output}", flush=True)
    if args.inventory_only:
        return
    for profile in ("full_clean", "random_perturbed_10s"):
        for index, row in enumerate(rows):
            seed = args.clean_seed if profile == "full_clean" else args.robust_seed
            report_path = output / f"{row['motion_id']}_{profile}.json"
            log_path = output / f"{row['motion_id']}_{profile}.log"
            signature = {"checkpoint": row["teacher"]["checkpoint_sha256"], "agent": row["teacher"]["agent_sha256"],
                         "env": row["env_yaml_sha256"], "motion": row["motion_sha256"], "implementation": implementation,
                         "profile": profile, "seed": seed, "num_envs": args.num_envs}
            signature_path = report_path.with_suffix(".signature.json")
            cached = args.resume and report_path.is_file() and signature_path.is_file() and json.loads(signature_path.read_text()) == signature
            print(f"START {profile} {index+1}/{len(rows)} {row['motion_id']} iteration={row['iteration']} cached={cached}", flush=True)
            command = [sys.executable, "-u", str(_bootstrap.ROOT / "scripts/stage2/rollout.py"), "--mode", "evaluate",
                       "--motion_file", row["motion_file"], "--teacher_checkpoint", row["checkpoint"],
                       "--agent_yaml", row["agent_yaml"], "--env_yaml", row["env_yaml"],
                       "--num_envs", str(args.num_envs), "--seed", str(seed), "--device", args.device,
                       "--output", str(report_path), "--headless", "--fast_exit", "--split", "test"]
            if profile == "random_perturbed_10s":
                command += ["--start_phase", "random", "--perturbed", "--evaluation_steps", "250"]
            row[profile] = {"status": "running", "command": command, "log": str(log_path)}
            summarize(output, manifest)
            code, error = 0, None
            if not cached:
                with open(log_path, "w") as log:
                    child = subprocess.Popen(command, cwd=_bootstrap.ROOT, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        code = child.wait(timeout=args.job_timeout)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                        code, error = -1, "Evaluation timed out; no quality verdict"
            if code or not report_path.is_file():
                row[profile].update(status="error", exit_code=code, error=error or "See evaluation log")
            else:
                report = json.loads(report_path.read_text())
                if report["teacher"] != row["teacher"] or report["motion_sha256"] != row["motion_sha256"]:
                    raise ValueError("Candidate changed during evaluation")
                atomic_json(signature_path, signature)
                row[profile].update({k: report[k] for k in ("successes", "trials", "success_rate", "passed_g1", "body_error_mean_m", "trial_ticks", "termination_terms", "termination_reasons", "wilson95")})
                row[profile].update(status="complete", report=str(report_path), report_sha256=file_hash(report_path))
                print(f"DONE {profile} {row['motion_id']}: {report['successes']}/{report['trials']} body_error={100*report['body_error_mean_m']:.2f}cm", flush=True)
            summarize(output, manifest)
    manifest["complete"] = True
    manifest["finished_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    summarize(output, manifest)
    print(f"All evaluations finished: {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
