"""Fixed observation and CVAE contract; no simulator imports."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn

SCHEMA = "beyondmimic-stage2-v1"
CONTRACT = {
    "schema": SCHEMA, "hz": 25, "reference_dim": 67, "proprio_dim": 96,
    "teacher_dim": 160, "action_dim": 29, "latent_dim": 32,
    "quaternion": "wxyz", "rot6d": "first_two_columns_row_major",
    "reference": "q_abs,qd,robot_anchor_position_error,robot_anchor_rotation_error",
    "proprio": "gravity_b,root_com_v_b,root_w_b,q_rel_runtime,qd_rel,previous_applied_action",
    "teacher": "reference,root_com_v_b,root_w_b,q_rel_runtime,qd_rel,previous_applied_action",
    "state_time": "pre_action", "action": "unclipped_normalized_actor_mean",
}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


CONTRACT_HASH = digest_json(CONTRACT)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def runtime_versions():
    from importlib.metadata import PackageNotFoundError, version
    result = {}
    for name in ("torch", "numpy", "isaaclab", "isaacsim", "rsl-rl-lib"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unavailable"
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def observations(ref_q, ref_qd, anchor_pos_b, anchor_rot6d, gravity_b, v_b, w_b,
                 q, qd, default_q, default_qd, previous_action):
    """Named fields, never a silent slice of the teacher vector."""
    reference = torch.cat((ref_q, ref_qd, anchor_pos_b, anchor_rot6d), -1)
    proprio = torch.cat((gravity_b, v_b, w_b, q - default_q, qd - default_qd, previous_action), -1)
    teacher = torch.cat((reference, v_b, w_b, q - default_q, qd - default_qd, previous_action), -1)
    for value, size in ((reference, 67), (proprio, 96), (teacher, 160)):
        if value.shape[-1] != size or not torch.isfinite(value).all():
            raise ValueError(f"Invalid observation {value.shape}, expected finite last dimension {size}")
    return reference, proprio, teacher


@dataclass
class ModelConfig:
    hidden: tuple = (2048, 1024, 512)
    latent_dim: int = 32
    beta: float = 0.01
    learning_rate: float = 5e-4
    accumulation: int = 15
    grad_clip: float = 1.0
    kl_reduction: str = "sum"
    logvar_min: float = -20.0
    logvar_max: float = 10.0


def mlp(inputs, hidden, outputs):
    layers = []
    for size in hidden:
        layers.extend((nn.Linear(inputs, size), nn.ELU()))
        inputs = size
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class FrozenStandardizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        return (x.float() - self.mean) / self.std.clamp_min(1e-3)

    def fit(self, batches):
        """Merge population moments in fp64; caller must supply D0 train only."""
        count, mean, m2 = 0, None, None
        for x in batches:
            x = x.detach().cpu().double()
            if not torch.isfinite(x).all() or len(x) == 0:
                raise ValueError("Nonfinite/empty statistics batch")
            n, bmean = len(x), x.mean(0)
            bm2 = ((x - bmean) ** 2).sum(0)
            if count == 0:
                mean, m2 = bmean, bm2
            else:
                delta = bmean - mean
                m2 = m2 + bm2 + delta.square() * count * n / (count + n)
                mean = mean + delta * n / (count + n)
            count += n
        if count == 0:
            raise ValueError("No D0 training rows for normalization")
        self.mean.copy_(mean.float())
        self.std.copy_((m2 / count).sqrt().float().clamp_min(1e-3))
        self.count.fill_(count)


class CVAE(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.latent_dim != 32 or self.config.kl_reduction not in ("sum", "mean"):
            raise ValueError("Stage2 contract requires latent32 and explicit KL reduction")
        self.reference_norm = FrozenStandardizer(67)
        self.proprio_norm = FrozenStandardizer(96)
        self.encoder = mlp(67, self.config.hidden, 64)
        self.decoder = mlp(128, self.config.hidden, 29)

    def encode(self, reference):
        mu, logvar = self.encoder(self.reference_norm(reference)).chunk(2, -1)
        # Numeric protection is explicit in checkpoint/code, not a KL schedule.
        return mu.float(), logvar.float().clamp(self.config.logvar_min, self.config.logvar_max)

    def decode(self, latent, proprio):
        return self.decoder(torch.cat((latent, self.proprio_norm(proprio)), -1))

    def forward(self, reference, proprio, sample=True):
        mu, logvar = self.encode(reference)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu) if sample else mu
        return self.decode(z, proprio), mu, logvar, z


def vae_loss(predicted, label, mu, logvar, beta=0.01, kl_reduction="sum"):
    rec = (predicted.float() - label.float()).square().mean()
    kl = 0.5 * (mu.float().square() + logvar.float().exp() - 1 - logvar.float())
    kl = (kl.sum(-1) if kl_reduction == "sum" else kl.mean(-1)).mean()
    return rec + beta * kl, rec, kl


class DecoderOnly(nn.Module):
    """Export API for stage3: exactly (latent_used, measured_proprio)."""
    def __init__(self, model):
        super().__init__()
        self.normalizer = model.proprio_norm
        self.decoder = model.decoder

    def forward(self, latent, proprio):
        return self.decoder(torch.cat((latent, self.normalizer(proprio)), -1))


def load_student(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("contract_hash") != CONTRACT_HASH:
        raise ValueError("Student checkpoint contract mismatch")
    model = CVAE(ModelConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


@torch.no_grad()
def label_and_action(snapshot, teacher, student=None, sample=False):
    """DAgger queries teacher on exactly the state on which student will act."""
    label = teacher(snapshot["teacher_observation"])
    if student is None:
        return label, label, {}
    action, mu, logvar, latent = student(snapshot["reference"], snapshot["proprio"], sample=sample)
    return label, action, {"mu": mu, "logvar": logvar, "latent_used": latent, "vae_action_clean": action}
