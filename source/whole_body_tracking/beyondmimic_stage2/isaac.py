"""Simulator adapter. Import ONLY after AppLauncher has started Isaac Sim."""
import copy
import json
from pathlib import Path
import tempfile

import gymnasium as gym
import torch
import yaml

from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
from isaaclab_tasks.utils import parse_env_cfg
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.observations import motion_anchor_ori_b, motion_anchor_pos_b
from .core import digest_json, file_hash, observations

TASK = "Tracking-Flat-G1-Low-Freq-v0"


class RecordedMotionCommand(MotionCommand):
    """Count every hidden teleport; evaluation clamps motion end without teleport."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.generation = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.in_update = False

    def _adaptive_sampling(self, env_ids):
        if getattr(self.cfg, "stage2_full_motion", False):
            if getattr(self.cfg, "stage2_start_phase", "zero") == "random":
                # Retain at least 5s to the end when the motion permits it.
                remaining = max(125, getattr(self.cfg, "stage2_min_remaining_frames", 125))
                self.time_steps[env_ids] = torch.randint(
                    0, max(1, self.motion.time_step_total - remaining), (len(env_ids),), device=self.device)
            else:
                self.time_steps[env_ids] = 0
        else:
            super()._adaptive_sampling(env_ids)

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.in_update and getattr(self.cfg, "stage2_full_motion", False):
            self.finished[env_ids] = True
            self.time_steps[env_ids] = self.motion.time_step_total - 1
            return
        self.generation[env_ids] += 1
        self.finished[env_ids] = False
        super()._resample_command(env_ids)

    def _update_command(self):
        self.in_update = True
        try:
            super()._update_command()
        finally:
            self.in_update = False

    def synchronize_targets(self):
        """Initialize relative reference after reset without advancing phase.

        The official command initializes relative targets to zero and updates
        them in compute() AFTER the first termination check. Full-motion eval
        must not misclassify that initialization artifact as a physical failure.
        """
        translation = self.robot_anchor_pos_w.clone()
        translation[:, 2] = self.anchor_pos_w[:, 2]
        rotation = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        rotation = rotation[:, None].expand(-1, len(self.cfg.body_names), -1)
        self.body_pos_relative_w = translation[:, None] + quat_apply(rotation, self.body_pos_w - self.anchor_pos_w[:, None])
        self.body_quat_relative_w = quat_mul(rotation, self.body_quat_w)


def check_teacher_environment(cfg, env_yaml):
    """Compare control, PD, robot initialization and policy observations before overrides.

    BaseLoader reads logged Python YAML tags as data without constructing objects.
    Paths and device-dependent runtime defaults are deliberately excluded.
    """
    saved = yaml.load(Path(env_yaml).read_text(), Loader=yaml.BaseLoader)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "current.yaml")
        dump_yaml(path, cfg)
        current = yaml.load(Path(path).read_text(), Loader=yaml.BaseLoader)
    checks = [(("decimation",),), (("sim", "dt"),), (("actions",),),
              (("observations", "policy"),), (("scene", "robot", "init_state"),),
              (("scene", "robot", "actuators"),), (("scene", "robot", "soft_joint_pos_limit_factor"),),
              (("scene", "robot", "spawn", "rigid_props"),), (("scene", "robot", "spawn", "articulation_props"),)]
    for (keys,) in checks:
        a, b = saved, current
        for key in keys:
            a, b = a[key], b[key]
        if a != b:
            raise ValueError(f"Teacher env.yaml differs from current task at {'.'.join(keys)}")
    for key in ("anchor_body_name", "body_names", "joint_position_range", "pose_range", "velocity_range"):
        if saved["commands"]["motion"][key] != current["commands"]["motion"][key]:
            raise ValueError(f"Teacher motion configuration differs at {key}")
    if saved["events"] != current["events"]:
        raise ValueError("Teacher domain randomization events differ from current task")


def create_environment(motion_file, num_envs, device, seed, env_yaml=None, full_motion=False, clean=False, start_phase="zero", min_remaining_frames=125):
    cfg = parse_env_cfg(TASK, device=device, num_envs=num_envs)
    if env_yaml:
        check_teacher_environment(cfg, env_yaml)
    cfg.seed = seed
    cfg.commands.motion.motion_file = str(Path(motion_file).resolve(strict=True))
    cfg.commands.motion.class_type = RecordedMotionCommand
    cfg.commands.motion.stage2_full_motion = full_motion
    cfg.commands.motion.stage2_start_phase = start_phase
    cfg.commands.motion.stage2_min_remaining_frames = min_remaining_frames
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    # Engineering choice: teacher labels and student both use the same clean state.
    # Physics randomization stays enabled for D0 and DAgger.
    cfg.observations.policy.enable_corruption = False
    if full_motion:
        import numpy as np
        with np.load(motion_file) as data:
            cfg.episode_length_s = len(data["joint_pos"]) / 25 + 10
    if clean:
        cfg.events.push_robot = None
        cfg.commands.motion.pose_range = {}
        cfg.commands.motion.velocity_range = {}
        cfg.commands.motion.joint_position_range = (0.0, 0.0)
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset(seed=seed)
    env.command_manager.get_term("motion").synchronize_targets()
    adapter = Adapter(env)
    return env, adapter


class Adapter:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.command = env.command_manager.get_term("motion")
        if len(self.robot.joint_names) != 29 or abs(env.step_dt - 0.04) > 1e-8:
            raise ValueError("Stage2 requires native G1 29DoF at 25Hz")
        if abs(float(self.command.motion.fps) - 25) > 1e-5:
            raise ValueError("Motion must be converted to 25Hz")
        self.ids = self.command.body_indexes
        action = env.action_manager.get_term("joint_pos")
        if getattr(action.cfg, "clip", None) is not None:
            raise ValueError("Clipped joint actions need a separate processed-action contract")
        action_ids = getattr(action, "_joint_ids")
        if not isinstance(action_ids, slice) and list(action_ids) != list(range(29)):
            raise ValueError("Action joint ordering differs from native articulation order")
        # Runtime randomized defaults are state, never part of invariant mapping_hash.
        scales = torch.as_tensor(action._scale).detach().cpu()
        if scales.ndim == 2:
            torch.testing.assert_close(scales, scales[0:1].expand_as(scales))
            scales = scales[0]
        scale = scales.reshape(-1).tolist()
        asset_root = Path(__file__).resolve().parent.parent / "whole_body_tracking"
        assets = [asset_root / "robots" / "g1.py", asset_root / "robots" / "actuator.py"]
        urdf = Path(env.cfg.scene.robot.spawn.asset_path)
        assets.append(urdf)
        assets.extend(sorted(p for p in (asset_root / "assets/unitree_description/meshes/g1").glob("*") if p.is_file()))
        self.mapping = {
            "joint_names": list(self.robot.joint_names), "body_names": list(self.command.cfg.body_names),
            "action_scale": scale, "nominal_init": copy.deepcopy(env.cfg.scene.robot.init_state.to_dict()),
            "assets": {str(p.relative_to(asset_root)): file_hash(p) for p in assets},
            "anchor": "torso_link", "root": "pelvis", "raw_velocity": "link_origin",
            "proprio_linear_velocity": "root_COM_body_matches_official_mdp",
            "actuator_config": env.cfg.scene.robot.actuators,
        }
        # Config objects include function/type objects; reuse audited YAML serialization.
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "actuators.yaml")
            dump_yaml(path, self.mapping["actuator_config"])
            self.mapping["actuator_config"] = yaml.load(Path(path).read_text(), Loader=yaml.BaseLoader)
        self.mapping_hash = digest_json(self.mapping)

    def snapshot(self):
        d, c, env = self.robot.data, self.command, self.env
        reference, proprio, teacher = observations(
            c.joint_pos, c.joint_vel, motion_anchor_pos_b(env, "motion"), motion_anchor_ori_b(env, "motion"),
            d.projected_gravity_b, d.root_lin_vel_b, d.root_ang_vel_b,
            d.joint_pos, d.joint_vel, d.default_joint_pos, d.default_joint_vel, env.action_manager.action,
        )
        actual = env.observation_manager.compute_group("policy")
        torch.testing.assert_close(teacher, actual, rtol=1e-5, atol=1e-5)
        # New Isaac API and audited legacy fallback have the same physical point.
        if hasattr(d, "body_link_lin_vel_w"):
            body_v = d.body_link_lin_vel_w
            root = d.root_link_state_w
        elif hasattr(d, "com_pos_b"):
            body_v = d.body_lin_vel_w + torch.cross(
                d.body_ang_vel_w, quat_apply(d.body_quat_w, -d.com_pos_b), dim=-1)
            root_v = d.root_lin_vel_w + torch.cross(
                d.root_ang_vel_w, quat_apply(d.root_quat_w, -d.com_pos_b[:, 0]), dim=-1)
            root = torch.cat((d.root_pos_w, d.root_quat_w, root_v, d.root_ang_vel_w), -1)
        else:
            raise RuntimeError("Cannot establish consistent link-origin velocities with this Isaac version")
        row = {
            "reference": reference, "proprio": proprio, "teacher_observation": teacher,
            "reference_frame": c.time_steps, "root_pos_w": root[:, :3], "root_quat_w": root[:, 3:7],
            "root_lin_vel_w": root[:, 7:10], "root_ang_vel_w": root[:, 10:13],
            "body_pos_w": d.body_pos_w[:, self.ids], "body_quat_w": d.body_quat_w[:, self.ids],
            "body_lin_vel_w": body_v[:, self.ids], "body_ang_vel_w": d.body_ang_vel_w[:, self.ids],
            "joint_pos": d.joint_pos, "joint_vel": d.joint_vel,
            "default_joint_pos_runtime": d.default_joint_pos, "default_joint_vel": d.default_joint_vel,
            "previous_action": env.action_manager.action, "generation": c.generation,
            "default_joint_pos_nominal": d.default_joint_pos_nominal.expand_as(d.joint_pos),
            "pd_action_offset": env.action_manager.get_term("joint_pos")._offset,
            "pd_target": env.action_manager.get_term("joint_pos").processed_actions,
        }
        return {k: v.detach().clone() for k, v in row.items()}

    def transition(self, action, capture_next=False):
        """Save terminal physics snapshot before auto-reset; do not substitute reset state."""
        manager = self.env.reward_manager
        original = manager.compute
        captured = {}

        def record_physics(*args, **kwargs):
            reward = original(*args, **kwargs)
            # Reward runs after physics/termination and BEFORE auto-reset/command update.
            captured["state"] = self.snapshot()
            captured["reasons"] = torch.stack([
                self.env.termination_manager.get_term(name).clone()
                for name in self.env.termination_manager.active_terms], -1)
            return reward

        manager.compute = record_physics
        try:
            _, reward, terminated, truncated, _ = self.env.step(action)
        finally:
            manager.compute = original
        if "state" not in captured:
            raise RuntimeError("Isaac step ordering changed: terminal physics snapshot was not captured")
        if capture_next:
            # Valid next_* means the state actually returned by the complete
            # control tick, including interval pushes. At a boundary preserve
            # the terminal physics snapshot and flag it invalid for continuity.
            following = self.snapshot()
            boundary = terminated | truncated | (self.command.generation != captured["state"]["generation"])
            captured["next"] = {}
            for key in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "body_pos_w",
                        "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "joint_pos", "joint_vel"):
                old, new = captured["state"][key], following[key]
                mask = boundary.reshape(-1, *([1] * (old.ndim - 1)))
                captured["next"][key] = torch.where(mask, old, new)
        return reward.clone(), terminated.clone(), truncated.clone(), captured

    def body_error(self, snapshot):
        """Official tracking yaw alignment, computed even on the first reset frame."""
        c = self.command
        translation = c.robot_anchor_pos_w.clone()
        translation[:, 2] = c.anchor_pos_w[:, 2]
        rotation = yaw_quat(quat_mul(c.robot_anchor_quat_w, quat_inv(c.anchor_quat_w)))
        target = translation[:, None] + quat_apply(
            rotation[:, None].expand(-1, len(c.cfg.body_names), -1), c.body_pos_w - c.anchor_pos_w[:, None])
        return torch.linalg.vector_norm(target - snapshot["body_pos_w"], dim=-1).mean(-1)
