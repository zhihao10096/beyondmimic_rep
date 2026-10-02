"""Train CVAE on real D0 / aggregated DAgger run directories. No Isaac import."""
import _bootstrap  # noqa: F401
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from beyondmimic_stage2.core import CONTRACT_HASH, CVAE, ModelConfig, atomic_json, load_student, runtime_versions, vae_loss
from beyondmimic_stage2.data import RolloutDataset


def validate(model, loader, device):
    model.eval()
    sums, count = np.zeros(3), 0
    with torch.no_grad():
        for ref, prop, label in loader:
            ref, prop, label = ref.to(device), prop.to(device), label.to(device)
            pred, mu, logvar, _ = model(ref, prop, sample=False)
            loss, rec, kl = vae_loss(pred, label, mu, logvar, model.config.beta, model.config.kl_reduction)
            sums += np.array([loss.item(), rec.item(), kl.item()]) * len(ref)
            count += len(ref)
    if not count:
        raise ValueError("A separate validation collector run is required")
    return dict(zip(("loss", "mse", "kl"), (sums / count).tolist()))


def optimizer_step(model, optimizer, rows):
    # Losses accumulated as row sums: uneven and partial accumulation windows are exact.
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(rows)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.config.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", nargs="+", required=True, help="Complete train AND val run directories")
    p.add_argument("--output", required=True)
    p.add_argument("--resume")
    p.add_argument("--new_round", action="store_true", help="Keep frozen D0 stats, start next dataset round")
    p.add_argument("--epochs", type=int, default=50, help="Engineering budget; not specified in paper")
    p.add_argument("--samples_per_epoch", type=int, default=100000)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--d0_fraction", type=float, default=0.25)
    p.add_argument("--allow_unqualified", action="store_true", help="Debug-only experiment")
    args = p.parse_args()
    if min(args.epochs, args.samples_per_epoch, args.batch_size) <= 0 or not 0 < args.d0_fraction < 1:
        p.error("Positive budgets and d0_fraction in (0,1) required")
    if args.new_round and not args.resume:
        p.error("--new_round requires --resume")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = RolloutDataset(args.data, "train", allow_unqualified=args.allow_unqualified)
    val = RolloutDataset(args.data, "val", allow_unqualified=args.allow_unqualified)
    if not len(train) or not len(val):
        raise ValueError("Need nonempty independent train and validation runs")
    if not (train.rounds == 0).any():
        raise ValueError("Aggregated training must retain D0")
    device = torch.device(args.device)
    model = CVAE().to(device)
    generator = torch.Generator().manual_seed(args.seed)
    start, step, best = 0, 0, float("inf")
    stats_source = train.fingerprint
    previous = None
    if args.resume:
        model, previous = load_student(args.resume, device)
        if previous["mapping_hash"] != train.mapping_hash:
            raise ValueError("Resume joint/action/asset mapping mismatch")
        if previous["allow_unqualified"] != args.allow_unqualified:
            raise ValueError("Cannot silently promote debug model into qualified experiment")
        if not args.new_round and (previous["train_fingerprint"] != train.fingerprint or
                                  previous["val_fingerprint"] != val.fingerprint):
            raise ValueError("Resume dataset changed; use --new_round explicitly")
        if not args.new_round:
            for key in ("batch_size", "samples_per_epoch", "seed", "d0_fraction"):
                if previous["training_options"][key] != getattr(args, key):
                    raise ValueError(f"Exact resume changed {key}; start --new_round for a new experiment")
        stats_source = previous["normalizer_d0_source"]
    else:
        if (train.rounds != 0).any():
            raise ValueError("Warmstart must be D0-only; subsequent DAgger requires --resume --new_round")
        model.reference_norm.fit(train.d0_batches("reference"))
        model.proprio_norm.fit(train.d0_batches("proprio"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.config.learning_rate, weight_decay=0)
    if previous:
        optimizer.load_state_dict(previous["optimizer"])
        step = previous["optimizer_step"]
        if not args.new_round:
            start, best = previous["epoch"] + 1, previous["best_val_mse"]
            torch.set_rng_state(previous["rng_torch"].cpu())
            if device.type == "cuda":
                torch.cuda.set_rng_state_all([x.cpu() for x in previous["rng_cuda"]])
            np.random.set_state(previous["rng_numpy"])
            random.setstate(previous["rng_python"])
            generator.set_state(previous["rng_sampler"].cpu())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and (
        not args.resume or args.new_round or Path(args.resume).resolve().parent != output.resolve()
    ):
        raise ValueError("Output already contains a checkpoint; resume that run or choose a new output for new_round")
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=0)
    sampler = train.sampler(args.samples_per_epoch, generator, args.d0_fraction)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    atomic_json(output / "run.json", {**vars(args), "config": asdict(model.config), "contract_hash": CONTRACT_HASH,
                                      "train_fingerprint": train.fingerprint, "val_fingerprint": val.fingerprint})
    for epoch in range(start, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rows, micro, clips = 0, 0, 0
        sums, count = np.zeros(3), 0
        for ref, prop, label in loader:
            ref, prop, label = ref.to(device), prop.to(device), label.to(device)
            pred, mu, logvar, _ = model(ref, prop, sample=True)
            loss, rec, kl = vae_loss(pred, label, mu, logvar, model.config.beta, model.config.kl_reduction)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite CVAE loss")
            (loss * len(ref)).backward()
            rows += len(ref)
            micro += 1
            sums += np.array([loss.item(), rec.item(), kl.item()]) * len(ref)
            count += len(ref)
            if micro == model.config.accumulation:
                clips += optimizer_step(model, optimizer, rows) > model.config.grad_clip
                step += 1
                rows, micro = 0, 0
        if rows:
            clips += optimizer_step(model, optimizer, rows) > model.config.grad_clip
            step += 1
        metrics = validate(model, val_loader, device)
        is_best = metrics["mse"] < best
        best = min(best, metrics["mse"])
        saved = {
            "contract_hash": CONTRACT_HASH, "config": asdict(model.config), "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": epoch, "optimizer_step": step, "best_val_mse": best,
            "train_fingerprint": train.fingerprint, "val_fingerprint": val.fingerprint,
            "normalizer_d0_source": stats_source, "mapping_hash": train.mapping_hash,
            "teacher_maps": [m["manifest"]["metadata"] for m in train.manifests],
            "allow_unqualified": args.allow_unqualified, "latent_convention": "posterior_mean_rollout",
            "training_options": {key: getattr(args, key) for key in ("batch_size", "samples_per_epoch", "seed", "d0_fraction")},
            "runtime_versions": runtime_versions(),
            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "rng_numpy": np.random.get_state(), "rng_python": random.getstate(), "rng_sampler": generator.get_state(),
        }
        for name in (("last.pt", "best.pt") if is_best else ("last.pt",)):
            tmp = output / (name + ".tmp")
            torch.save(saved, tmp)
            os.replace(tmp, output / name)
        record = {"epoch": epoch, "step": step, "train": (sums / count).tolist(), "validation": metrics, "gradient_clips": clips}
        with open(output / "metrics.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
