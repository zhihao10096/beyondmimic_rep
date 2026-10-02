"""Export a reference-free TorchScript decoder for stage3."""
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import torch
from beyondmimic_stage2.core import CONTRACT, DecoderOnly, atomic_json, file_hash, load_student

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
model, saved = load_student(args.checkpoint)
decoder = DecoderOnly(model).eval()
z, proprio = torch.randn(7, 32), torch.randn(7, 96)
with torch.no_grad():
    exported = torch.jit.trace(decoder, (z, proprio))
    torch.testing.assert_close(exported(z, proprio), model.decode(z, proprio))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    exported.save(str(path))
    reloaded = torch.jit.load(str(path))
    torch.testing.assert_close(reloaded(z[:1], proprio[:1]), model.decode(z[:1], proprio[:1]))
atomic_json(str(path) + ".json", {"contract": CONTRACT, "source_sha256": file_hash(args.checkpoint),
                                  "mapping_hash": saved["mapping_hash"], "inputs": ["latent32", "measured_proprio96"],
                                  "allow_unqualified": saved["allow_unqualified"]})
print(f"Decoder exported and reloaded: {path}")
