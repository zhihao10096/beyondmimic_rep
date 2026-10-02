"""Immutable run manifests, bounded NPZ shards and balanced lazy sampling."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .core import CONTRACT_HASH, SCHEMA, atomic_json, digest_json, file_hash


def mmap_npz(path, key):
    """Map an uncompressed NPZ member directly; random minibatches never unpack a shard.

    np.savez writes ZIP_STORED NPY entries. Local header lengths locate the NPY
    header including ZIP64 extras; numpy parses dtype/shape without pickle.
    """
    import struct
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError("Stage2 shards must be uncompressed np.savez for bounded-memory random access")
    with open(path, "rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        stream.seek(name_len + extra_len, 1)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY format {version}")
        if dtype.hasobject:
            raise ValueError("Object arrays are forbidden")
        offset = stream.tell()
    return np.memmap(path, mode="r", dtype=dtype, shape=shape, offset=offset, order="F" if fortran else "C")


class ShardWriter:
    def __init__(self, output, metadata, shard_rows=8192):
        self.output = Path(output)
        # A run directory belongs to one writer; never append over another run.
        self.output.mkdir(parents=True, exist_ok=False)
        if metadata["split"] not in ("train", "val", "test"):
            raise ValueError("Choose a complete-run split")
        if shard_rows <= 0:
            raise ValueError("shard_rows must be positive")
        self.manifest = {"schema": SCHEMA, "contract_hash": CONTRACT_HASH, "metadata": metadata, "shards": []}
        self.limit, self.pending, self.count, self.fields = shard_rows, [], 0, None
        self._publish()

    def _publish(self):
        atomic_json(self.output / "manifest.json", self.manifest)

    def append(self, batch):
        batch = {k: v.detach().cpu().numpy().copy() if torch.is_tensor(v) else np.asarray(v).copy()
                 for k, v in batch.items()}
        sizes = {len(v) for v in batch.values()}
        if len(sizes) != 1:
            raise ValueError("Inconsistent batch lengths")
        if self.fields is None:
            self.fields = {k: (v.shape[1:], str(v.dtype)) for k, v in batch.items()}
        if self.fields != {k: (v.shape[1:], str(v.dtype)) for k, v in batch.items()}:
            raise ValueError("Changing row fields/shapes within a run")
        for name, v in batch.items():
            if v.dtype.kind == "O" or (v.dtype.kind == "f" and not np.isfinite(v).all()):
                raise ValueError(f"Nonfinite/object field: {name}")
        for start in range(0, next(iter(sizes)), self.limit):
            part = {k: v[start:start + self.limit] for k, v in batch.items()}
            self.pending.append(part)
            self.count += len(next(iter(part.values())))
            if self.count >= self.limit:
                self.flush()

    def flush(self):
        if not self.pending:
            return
        arrays = {k: np.concatenate([b[k] for b in self.pending]) for k in self.pending[0]}
        name = f"shard_{len(self.manifest['shards']):06d}.npz"
        path = self.output / name
        with open(path.with_suffix(".tmp"), "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path.with_suffix(".tmp"), path)
        self.manifest["shards"].append({"path": name, "rows": self.count, "sha256": file_hash(path)})
        self.manifest["fields"] = {k: {"shape": list(v.shape[1:]), "dtype": str(v.dtype)} for k, v in arrays.items()}
        self._publish()
        self.pending, self.count = [], 0

    def close(self):
        self.flush()
        self.manifest["complete"] = True
        self._publish()


class RolloutDataset(Dataset):
    def __init__(self, roots, split, cache_shards=64, allow_unqualified=False):
        self.shards, self.manifests, self.ends = [], [], []
        self.cache, self.cache_size = OrderedDict(), cache_shards
        self.groups, self.rounds = [], []
        total = 0
        identities = set()
        seed_splits = {}
        mapping = None
        for root in roots:
            root = Path(root)
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["schema"] != SCHEMA or manifest["contract_hash"] != CONTRACT_HASH:
                raise ValueError("Dataset contract mismatch")
            if not manifest.get("complete"):
                raise ValueError(f"Incomplete collector run: {root}")
            meta = manifest["metadata"]
            if not meta.get("teacher_qualified") and not allow_unqualified:
                raise ValueError("Unqualified teacher data is debug-only; use explicit --allow_unqualified")
            identity = (meta["motion_id"], meta["collector_seed"], meta["round"])
            if identity in identities:
                raise ValueError("Duplicate motion/collector seed/round across partitions or input roots")
            identities.add(identity)
            seed_key = identity[:2]
            if seed_key in seed_splits and seed_splits[seed_key] != meta["split"]:
                raise ValueError("Collector seed crosses train/val/test partitions across rounds")
            seed_splits[seed_key] = meta["split"]
            if mapping is None:
                mapping = meta["mapping_hash"]
            if mapping != meta["mapping_hash"]:
                raise ValueError("Joint/action/asset mapping mismatch")
            if meta["split"] != split:
                continue
            self.manifests.append({"root": str(root), "manifest": manifest})
            for shard in manifest["shards"]:
                path = (root / shard["path"]).resolve()
                if not path.is_relative_to(root.resolve()) or file_hash(path) != shard["sha256"]:
                    raise ValueError(f"Invalid shard path/hash: {path}")
                with np.load(path, allow_pickle=False) as data:
                    n = shard["rows"]
                    for key, dim in (("reference", 67), ("proprio", 96), ("teacher_label", 29)):
                        if data[key].shape != (n, dim) or not np.isfinite(data[key]).all():
                            raise ValueError(f"Invalid {key} in {path}")
                    phases = (data["reference_frame"] * 20 // max(1, meta["motion_frames"])).clip(0, 19)
                    self.groups.extend((meta["motion_id"], meta["round"], int(p)) for p in phases)
                    self.rounds.extend([meta["round"]] * n)
                total += n
                self.shards.append(path)
                self.ends.append(total)
        self.ends = np.asarray(self.ends, dtype=np.int64)
        self.rounds = np.asarray(self.rounds, dtype=np.int32)
        self.mapping_hash = mapping
        # Portable provenance excludes local paths.
        self.fingerprint = digest_json([m["manifest"] for m in self.manifests])

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def shard(self, index):
        if index not in self.cache:
            self.cache[index] = {k: mmap_npz(self.shards[index], k) for k in ("reference", "proprio", "teacher_label")}
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return self.cache[index]

    def __getitem__(self, index):
        shard = int(np.searchsorted(self.ends, index, side="right"))
        offset = index - (int(self.ends[shard - 1]) if shard else 0)
        return tuple(torch.from_numpy(self.shard(shard)[k][offset].copy())
                     for k in ("reference", "proprio", "teacher_label"))

    def d0_batches(self, key):
        for index, shard in enumerate(self.shards):
            start = int(self.ends[index - 1]) if index else 0
            if self.rounds[start] == 0:
                yield torch.from_numpy(np.array(self.shard(index)[key], copy=True))

    def sampler(self, samples, generator, d0_fraction=0.25):
        from collections import Counter
        counts = Counter(self.groups)
        weights = np.array([1 / counts[g] for g in self.groups], dtype=np.float64)
        d0 = self.rounds == 0
        if d0.any() and (~d0).any():
            weights[d0] *= d0_fraction / weights[d0].sum()
            weights[~d0] *= (1 - d0_fraction) / weights[~d0].sum()
        return WeightedRandomSampler(torch.from_numpy(weights), samples, replacement=True, generator=generator)
