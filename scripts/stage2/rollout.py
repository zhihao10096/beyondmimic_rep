"""Isaac entry point: adapter smoke, real D0/DAgger collection, full-motion evaluation."""
import _bootstrap
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from isaaclab.app import AppLauncher


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("adapter", "collect", "evaluate"), required=True)
    p.add_argument("--motion_file", required=True)
    p.add_argument("--teacher_checkpoint")
    p.add_argument("--agent_yaml")
    p.add_argument("--env_yaml")
    p.add_argument("--student_checkpoint")
    p.add_argument("--teacher_gate", help="G1 full-motion evaluation JSON from this exact teacher/motion")
    p.add_argument("--output", required=True, help="Fresh run directory for collect; JSON file for other modes")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=2500, help="Collector ticks per environment; adapter ticks")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--split", choices=("train", "val", "test"), default="train")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--shard_rows", type=int, default=8192)
    p.add_argument("--latent", choices=("mean", "sample"), default="mean")
    p.add_argument("--perturbed", action="store_true", help="Eval: retain reset noise and interval pushes")
    p.add_argument("--start_phase", choices=("zero", "random"), default="zero", help="Evaluation only; random start keeps >=5s when possible")
    p.add_argument("--evaluation_steps", type=int, default=0, help="Diagnostic evaluation horizon; 0=motion end. A positive horizon never qualifies G1")
    p.add_argument("--allow_unqualified", action="store_true", help="Explicit debug D0/DAgger only")
    p.add_argument("--fast_exit", action="store_true", help="Headless: exit after verified writes, bypass Kit close hang")
    AppLauncher.add_app_launcher_args(p)
    p.set_defaults(device="cuda:0")
    return p


def preflight(p, args):
    for name in ("motion_file", "teacher_checkpoint", "agent_yaml", "env_yaml", "student_checkpoint", "teacher_gate"):
        value = getattr(args, name)
        if value and not Path(value).is_file():
            p.error(f"{name} does not exist: {value}")
    if min(args.num_envs, args.steps, args.shard_rows) <= 0:
        p.error("num_envs, steps and shard_rows must be positive")
    if args.mode == "collect":
        if not all((args.teacher_checkpoint, args.agent_yaml, args.env_yaml)):
            p.error("Collect requires teacher_checkpoint, agent_yaml and env_yaml")
        if (args.round == 0) != (args.student_checkpoint is None) or args.round < 0:
            p.error("D0: round=0, no student. DAgger: round>=1 and student required")
        if not args.teacher_gate and not args.allow_unqualified:
            p.error("Run teacher full-motion evaluation first; qualified G1 JSON required")
        if Path(args.output).exists():
            p.error("Collection output already exists; choose a new complete-run directory")
        if args.split == "test":
            p.error("Test trajectories are evaluation only; collect train/val independently")
    if args.mode == "evaluate" and not args.student_checkpoint and not all((args.teacher_checkpoint, args.agent_yaml, args.env_yaml)):
        p.error("Teacher evaluation requires checkpoint and original agent/env YAML")
    if args.student_checkpoint and not args.env_yaml:
        p.error("Student rollout requires original env_yaml for physics contract verification")
    if args.fast_exit and not args.headless:
        p.error("--fast_exit is only for headless runs")
    if args.start_phase != "zero" and args.mode != "evaluate":
        p.error("--start_phase controls evaluation only; collectors retain official adaptive sampling")
    if args.evaluation_steps < 0 or (args.evaluation_steps and args.mode != "evaluate"):
        p.error("--evaluation_steps must be nonnegative and is evaluation-only")


def main(args, app):
    import numpy as np
    import torch
    from beyondmimic_stage2.core import CONTRACT_HASH, atomic_json, file_hash, label_and_action, load_student, runtime_versions
    from beyondmimic_stage2.data import ShardWriter
    from beyondmimic_stage2.isaac import TASK, create_environment
    from beyondmimic_stage2.teacher import Teacher

    teacher = Teacher(args.teacher_checkpoint, args.agent_yaml, args.device) if args.teacher_checkpoint else None
    student, checkpoint = load_student(args.student_checkpoint, args.device) if args.student_checkpoint else (None, None)
    env, adapter = create_environment(args.motion_file, args.num_envs, args.device, args.seed, args.env_yaml,
                                      full_motion=args.mode == "evaluate",
                                      clean=args.mode == "evaluate" and not args.perturbed, start_phase=args.start_phase,
                                      min_remaining_frames=max(125, args.evaluation_steps))
    motion_hash = file_hash(args.motion_file)
    qualified = False
    gate = json.loads(Path(args.teacher_gate).read_text()) if args.teacher_gate else None
    if gate:
        identity_matches = (gate.get("motion_sha256") == motion_hash
                     and gate.get("mapping_hash") == adapter.mapping_hash
                     and gate.get("teacher") == (teacher.metadata if teacher else None)
                     and gate.get("env_yaml_sha256") == (file_hash(args.env_yaml) if args.env_yaml else None)
                     and gate.get("contract_hash") == CONTRACT_HASH)
        if not identity_matches:
            raise ValueError("Teacher gate belongs to a different teacher/motion/mapping")
        qualified = gate.get("passed_g1") is True
        if not qualified and not args.allow_unqualified:
            raise ValueError("Teacher gate failed; only explicit --allow_unqualified debug runs are permitted")
    if args.mode == "collect" and not qualified and not args.allow_unqualified:
        raise ValueError("Only G1-qualified teachers can enter formal D0/DAgger")
    if student:
        if checkpoint["mapping_hash"] != adapter.mapping_hash:
            raise ValueError("Student checkpoint joint/action/asset mapping mismatch")
        if checkpoint["allow_unqualified"] and not args.allow_unqualified:
            raise ValueError("Debug student requires --allow_unqualified")
    metadata = {
        "task": TASK, "contract_hash": CONTRACT_HASH, "mapping": adapter.mapping,
        "mapping_hash": adapter.mapping_hash, "motion_id": Path(args.motion_file).stem,
        "motion_sha256": motion_hash, "motion_frames": adapter.command.motion.time_step_total,
        "collector_seed": args.seed, "split": args.split, "round": args.round,
        "num_envs": args.num_envs, "control_dt": env.step_dt,
        "executed_policy": "student_only" if student else "teacher_mean" if teacher else "zero_action_interface_smoke",
        "teacher": teacher.metadata if teacher else None, "teacher_qualified": qualified,
        "teacher_gate_sha256": file_hash(args.teacher_gate) if args.teacher_gate else None,
        "env_yaml_sha256": file_hash(args.env_yaml) if args.env_yaml else None,
        "student_sha256": file_hash(args.student_checkpoint) if student else None,
        "latent": args.latent if student else None, "observation_noise": "disabled_clean_named_state",
        "physics_randomization": "original_startup_and_interval" if args.mode != "evaluate" or args.perturbed else "startup_only_zero_reset_noise_no_push",
        "termination_terms": list(env.termination_manager.active_terms),
        "initial_reference_frames": adapter.command.time_steps.cpu().tolist(),
        "env_origins_w": env.scene.env_origins.cpu().tolist(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_bootstrap.ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=_bootstrap.ROOT, text=True).strip()),
        "runtime_versions": runtime_versions(),
        "next_state_timing": "returned_state_including_interval_events_if_valid_else_terminal_physics_before_reset",
        "implementation_sha256": {str(p.relative_to(_bootstrap.ROOT)): file_hash(p)
                                   for p in (_bootstrap.ROOT / "source/whole_body_tracking/beyondmimic_stage2").glob("*.py")},
    }
    if args.mode == "adapter":
        max_offset_difference = (adapter.robot.data.default_joint_pos - env.action_manager.get_term("joint_pos")._offset).abs().max().item()
        for _ in range(args.steps):
            snapshot = adapter.snapshot()
            torch.testing.assert_close(snapshot["previous_action"], env.action_manager.action)
            # Interface smoke only, no fabricated teacher data/checkpoint.
            adapter.transition(torch.zeros(args.num_envs, 29, device=args.device))
        command = adapter.command
        command.cfg.stage2_full_motion = True
        command.time_steps[:] = command.motion.time_step_total - 1
        generation, root_before = command.generation.clone(), adapter.robot.data.root_state_w.clone()
        command._update_command()
        assert command.finished.all() and (command.time_steps == command.motion.time_step_total - 1).all()
        torch.testing.assert_close(command.generation, generation)
        torch.testing.assert_close(adapter.robot.data.root_state_w, root_before)
        command._resample_command(torch.arange(args.num_envs, device=args.device))
        torch.testing.assert_close(command.generation, generation + 1)
        assert not command.finished.any() and (command.time_steps == 0).all()
        command.cfg.stage2_start_phase = "random"
        command._resample_command(torch.arange(args.num_envs, device=args.device))
        assert (command.time_steps >= 0).all() and (command.time_steps < max(1, command.motion.time_step_total - 125)).all()
        atomic_json(args.output, {"passed": True, "checks": ["clean_teacher160_exact", "reference67", "proprio96",
                                                             "pre_and_post_physics", "link_origin_velocities", "full_motion_end_no_teleport", "resample_generation", "random_phase_reset"],
                                 "metadata": metadata, "steps": args.steps, "max_runtime_default_vs_pd_offset": max_offset_difference})
        print(f"Adapter smoke passed: {args.output}", flush=True)
    elif args.mode == "collect":
        writer = ShardWriter(args.output, metadata, args.shard_rows)
        episodes = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        env_ids = torch.arange(args.num_envs, device=args.device)
        for tick in range(args.steps):
            with torch.no_grad():
                row = adapter.snapshot()
                label, executed, latent_fields = label_and_action(row, teacher, student, sample=args.latent == "sample")
                row.update(latent_fields)
                # Both policies queried from this ONE snapshot; only student executes in DAgger.
                reward, terminated, truncated, captured = adapter.transition(executed, capture_next=True)
                resampled = adapter.command.generation != row["generation"]
                reset = terminated | truncated
                boundary = reset | resampled
                row.update(
                    teacher_label=label, action_executed=executed, action_proposed=executed,
                    pd_target=captured["state"]["pd_target"],
                    reward=reward, terminated=terminated, truncated=truncated, reset=reset,
                    command_resampled=resampled, termination_reason=captured["reasons"],
                    env_id=env_ids, episode_id=episodes.clone(), next_episode_id=episodes + boundary.long(),
                    control_step=torch.full_like(episodes, tick), time_s=torch.full((args.num_envs,), tick * env.step_dt, device=args.device),
                    next_state_valid=~boundary,
                )
                for key in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "body_pos_w",
                            "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
                    row["next_" + key] = captured["next"][key]
                writer.append(row)
                episodes += boundary.long()
            if (tick + 1) % 100 == 0:
                print(f"Collected {tick + 1}/{args.steps} ticks, {args.num_envs * (tick + 1)} rows", flush=True)
            if not app.is_running():
                raise RuntimeError("Simulator interrupted: manifest deliberately remains incomplete")
        writer.close()
        print(f"Complete real {'DAgger' if student else 'D0'} run: {args.output}", flush=True)
    else:
        active = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        success = torch.zeros_like(active)
        lengths = torch.zeros(args.num_envs, dtype=torch.long, device=args.device)
        body_sums = torch.zeros(args.num_envs, device=args.device)
        reasons = torch.zeros(args.num_envs, len(metadata["termination_terms"]), dtype=torch.bool, device=args.device)
        evaluation_limit = args.evaluation_steps or adapter.command.motion.time_step_total + 2
        for tick in range(evaluation_limit):
            with torch.no_grad():
                row = adapter.snapshot()
                error = adapter.body_error(row)
                body_sums += error * active
                lengths += active.long()
                action = student(row["reference"], row["proprio"], sample=args.latent == "sample")[0] if student else teacher(row["teacher_observation"])
                _, terminated, truncated, captured = adapter.transition(action)
                failure = (terminated | truncated) & active
                reasons[failure] = captured["reasons"][failure]
                completed = adapter.command.finished & active & ~failure
                success |= completed
                active &= ~(failure | completed)
            if not active.any():
                break
            if (tick + 1) % 250 == 0:
                print(f"Evaluation tick={tick + 1}, active={int(active.sum())}/{args.num_envs}, success={int(success.sum())}", flush=True)
        survived_horizon = active.clone() if args.evaluation_steps else torch.zeros_like(active)
        success |= survived_horizon
        rate = int(success.sum()) / args.num_envs
        # Wilson 95% interval: always include denominator and failed trials.
        n, z = args.num_envs, 1.96
        center = (rate + z*z/(2*n)) / (1+z*z/n)
        half = z * ((rate*(1-rate)/n + z*z/(4*n*n)) ** 0.5) / (1+z*z/n)
        protocol = f"{'full_motion_phase0' if args.start_phase == 'zero' else 'random_phase_to_end'}_{'perturbed' if args.perturbed else 'clean'}"
        if args.evaluation_steps:
            protocol = f"{'phase0' if args.start_phase == 'zero' else 'random_phase'}_horizon{args.evaluation_steps}_ticks_{'perturbed' if args.perturbed else 'clean'}"
        report = {**metadata, "policy": "student_only" if student else "teacher_mean", "protocol": protocol,
                  "trials": n, "successes": int(success.sum()), "success_rate": rate, "wilson95": [max(0.0, center-half), min(1.0, center+half)],
                  "body_error_mean_m": (body_sums / lengths.clamp_min(1)).mean().item(),
                  "trial_success": success.cpu().tolist(), "trial_ticks": lengths.cpu().tolist(),
                  "termination_reasons": reasons.cpu().tolist(), "unfinished": int(active.sum()) if not args.evaluation_steps else 0,
                  "evaluation_steps": args.evaluation_steps, "survived_horizon": survived_horizon.cpu().tolist(),
                  "reference_completed": (success & ~survived_horizon).cpu().tolist(),
                  "success_definition": "reference_end_or_survived_diagnostic_horizon" if args.evaluation_steps else "reference_end_without_physical_failure",
                  "passed_g1": bool(not student and not args.evaluation_steps and not args.perturbed and args.start_phase == "zero" and n >= 20 and rate >= .95)}
        if student and gate and report["protocol"] == gate["protocol"]:
            report["baseline_comparable"] = bool(gate["collector_seed"] == args.seed and gate["num_envs"] == n and args.latent == "mean")
            report["passed_g2_clean"] = bool(report["baseline_comparable"] and n >= 20 and rate >= .90 and gate["success_rate"] - rate <= .10 + 1e-8
                                              and report["body_error_mean_m"] <= 1.25 * gate["body_error_mean_m"]
                                              and not checkpoint["allow_unqualified"])
        atomic_json(args.output, report)
        print(json.dumps({k: report[k] for k in ("policy", "protocol", "trials", "successes", "success_rate", "passed_g1", "body_error_mean_m")}, ensure_ascii=False), flush=True)
    return env


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    preflight(p, args)
    launcher = AppLauncher(args)
    # Exceptions produce nonzero exit and never publish a complete collector manifest.
    try:
        env = main(args, launcher.app)
    except BaseException as error:
        if args.fast_exit:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130 if isinstance(error, KeyboardInterrupt) else 1)
        raise
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    env.close()
    launcher.app.close()
