"""Audit pinned G1 motion CSV/NPZ assets without importing Isaac Sim.

This checks kinematic file integrity, not teacher quality or physical feasibility.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_joint_names(project_root: Path) -> list[str]:
    tree = ast.parse((project_root / "scripts/csv_to_npz.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "run_simulator":
            for kw in node.keywords:
                if kw.arg == "joint_names":
                    return ast.literal_eval(kw.value)
    raise ValueError("Could not find the converter's explicit CSV joint mapping")


def audit_csv(path: Path, expected_sha: str, names: list[str], limits: dict) -> dict:
    if sha256(path) != expected_sha:
        raise ValueError(f"Source hash mismatch: {path}")
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 36 or len(data) < 3 or not np.isfinite(data).all():
        raise ValueError(f"Expected finite [T,36] CSV with at least 3 frames: {path}")
    quat = data[:, 3:7]
    norm = np.linalg.norm(quat, axis=1)
    if np.max(np.abs(norm - 1)) > 1e-3:
        raise ValueError(f"Invalid XYZW quaternion norms: {path}")
    quat = quat / norm[:, None]
    tilt = np.arccos(np.clip(1 - 2 * (quat[:, 0] ** 2 + quat[:, 1] ** 2), -1, 1))
    speed = np.linalg.norm(np.gradient(data[:, :3], 1 / 30, axis=0), axis=1)
    root_step = np.linalg.norm(np.diff(data[:, :3], axis=0), axis=1)
    joint_step = np.abs(np.diff(data[:, 7:], axis=0))
    limit_warnings = []
    for i, name in enumerate(names):
        if name in limits:
            low, high = limits[name]
            outside = (data[:, 7 + i] < low - 1e-3) | (data[:, 7 + i] > high + 1e-3)
            if outside.any():
                limit_warnings.append({"joint": name, "fraction": float(outside.mean())})
    return {
        "frames": len(data), "fps": 30, "duration_s": (len(data) - 1) / 30,
        "root_height_range_m": [float(data[:, 2].min()), float(data[:, 2].max())],
        "root_speed_p99_m_s": float(np.quantile(speed, .99)),
        "root_speed_max_m_s": float(speed.max()),
        "root_step_max_m": float(root_step.max()),
        "joint_step_max_rad": float(joint_step.max()),
        "root_tilt_max_deg": float(np.rad2deg(tilt.max())),
        "low_root_fraction_below_035m": float((data[:, 2] < .35).mean()),
        "urdf_joint_limit_warnings": limit_warnings,
        "note": "Height/tilt are kinematic indicators; they do not identify exact skills or prove a policy can execute them.",
    }


def audit_npz(path: Path) -> dict:
    keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    with np.load(path, allow_pickle=False) as saved:
        fps = np.asarray(saved["fps"]).reshape(-1)
        if fps.size != 1 or fps[0] != 25:
            raise ValueError(f"Expected 25Hz motion: {path}")
        arrays = {key: saved[key] for key in keys}
        n = len(arrays["joint_pos"])
        body_count = arrays["body_pos_w"].shape[1]
        shapes = {"joint_pos": (n, 29), "joint_vel": (n, 29), "body_pos_w": (n, body_count, 3),
                  "body_quat_w": (n, body_count, 4), "body_lin_vel_w": (n, body_count, 3),
                  "body_ang_vel_w": (n, body_count, 3)}
        if n < 3:
            raise ValueError(f"Motion is too short: {path}")
        for key, array in arrays.items():
            if array.shape != shapes[key] or not np.isfinite(array).all():
                raise ValueError(f"Incomplete or invalid NPZ field {key}: {path}")
        quat = arrays["body_quat_w"]
        if np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1)) > 1e-3:
            raise ValueError(f"Invalid NPZ body quaternions: {path}")
        return {"fps": 25, "frames": n, "body_count": body_count, "sha256": sha256(path),
                "bytes": path.stat().st_size, "shapes": {k: list(a.shape) for k, a in arrays.items()},
                "body_origin_height_min_m": float(arrays["body_pos_w"][..., 2].min()),
                "body_origin_height_note": "Link origins are not collision surfaces; this is not a penetration test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw_dir", type=Path, default=Path("data/reproduction/raw_g1"))
    parser.add_argument("--npz_dir", type=Path, default=None, help="If provided, all corresponding NPZ files must exist.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected_report", type=Path, default=None,
                        help="Verify copied NPZ hashes and URDF against the original preparation report.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = csv_joint_names(root)
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("CSV mapping must have exactly 29 distinct joint names")
    urdf = root / "source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
    limits = {}
    if urdf.exists():
        for joint in ET.parse(urdf).getroot().findall("joint"):
            lim = joint.find("limit")
            if lim is not None and "lower" in lim.attrib and "upper" in lim.attrib:
                limits[joint.attrib["name"]] = (float(lim.attrib["lower"]), float(lim.attrib["upper"]))
    entries = json.loads(args.manifest.read_text())
    expected = None
    if args.expected_report is not None:
        original = json.loads(args.expected_report.read_text())
        expected = {entry["motion_id"]: entry for entry in original["motions"]}
        if args.npz_dir is not None and (not urdf.exists() or sha256(urdf) != original["urdf_sha256"]):
            raise ValueError("Local URDF differs from the prepared NPZ asset; resolve the asset mismatch first")
    records = []
    for entry in entries:
        mid = entry["motion_id"]
        record = {"motion_id": mid, "csv": audit_csv(args.raw_dir / f"{mid}.csv", entry["sha256"], names, limits)}
        if args.npz_dir is not None:
            record["npz"] = audit_npz(args.npz_dir / f"{mid}.npz")
            if expected is not None and record["npz"]["sha256"] != expected[mid]["npz"]["sha256"]:
                raise ValueError(f"Copied NPZ hash mismatch: {mid}")
            # Float32 arange in the converter may differ by at most one endpoint.
            nominal = record["csv"]["duration_s"] * 25
            if abs(record["npz"]["frames"] - nominal) > 1.01:
                raise ValueError(f"NPZ duration does not match full CSV: {mid}")
        records.append(record)
        print(f"[OK] {mid}: {record['csv']['frames']} CSV frames", flush=True)
    report = {"scope": "kinematic assets only; no teacher quality claim", "csv_joint_names": names,
              "urdf_sha256": sha256(urdf) if urdf.exists() else None, "joint_limit_check_available": bool(limits),
              "motions": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
