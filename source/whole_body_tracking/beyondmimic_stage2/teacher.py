"""Strict frozen RSL-RL actor and its original empirical normalizer."""
from pathlib import Path
import torch
from torch import nn
import yaml

from .core import file_hash, mlp


class Teacher(nn.Module):
    def __init__(self, checkpoint, agent_yaml, device="cpu", normalizer_eps=1e-2):
        super().__init__()
        # SafeLoader rejects Python object tags; agent.yaml is a primitive mapping.
        cfg = yaml.safe_load(Path(agent_yaml).read_text())
        policy = cfg["policy"]
        if policy["activation"].lower() != "elu" or policy.get("class_name", "ActorCritic") != "ActorCritic":
            raise ValueError("Only audited RSL ActorCritic ELU teachers are supported")
        if cfg.get("clip_actions") is not None:
            raise ValueError("Clipped teacher needs a separate processed-action contract")
        self.actor = mlp(160, policy["actor_hidden_dims"], 29)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = {k.removeprefix("actor."): v for k, v in saved["model_state_dict"].items() if k.startswith("actor.")}
        self.actor.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in self.actor.state_dict().values()):
            raise ValueError("Teacher actor contains nonfinite weights")
        self.register_buffer("mean", torch.zeros(1, 160))
        self.register_buffer("std", torch.ones(1, 160))
        self.eps = float(normalizer_eps)
        self.normalized = bool(cfg["empirical_normalization"])
        if self.normalized:
            norm = saved.get("obs_norm_state_dict")
            if norm is None or set(norm) != {"_mean", "_std", "_var", "count"}:
                raise ValueError("Missing/unsupported original RSL observation normalizer")
            for name in ("_mean", "_std", "_var"):
                if norm[name].shape != (1, 160) or not torch.isfinite(norm[name]).all():
                    raise ValueError("Teacher normalizer is invalid")
            if (norm["_std"] < 0).any() or int(norm["count"]) <= 0:
                raise ValueError("Teacher normalizer has no fitted observations")
            self.mean.copy_(norm["_mean"])
            self.std.copy_(norm["_std"])
        self.metadata = {
            "checkpoint_sha256": file_hash(checkpoint), "agent_sha256": file_hash(agent_yaml),
            "empirical_normalization": self.normalized, "normalizer_eps": self.eps,
            "actor_hidden_dims": policy["actor_hidden_dims"], "label": "actor_mean",
        }
        self.to(device).eval().requires_grad_(False)

    def forward(self, observation):
        if observation.shape[-1] != 160:
            raise ValueError("Teacher expects clean raw 160-dimensional observations")
        x = (observation - self.mean) / (self.std + self.eps) if self.normalized else observation
        return self.actor(x)
