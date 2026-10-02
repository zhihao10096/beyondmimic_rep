"""Numerical/schema tests only. Synthetic fixtures never become rollout data."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/whole_body_tracking"))
from beyondmimic_stage2.core import (CONTRACT_HASH, CVAE, DecoderOnly, FrozenStandardizer, ModelConfig,
                                    label_and_action, load_student, observations, vae_loss)
from beyondmimic_stage2.data import RolloutDataset, ShardWriter, mmap_npz
from beyondmimic_stage2.teacher import Teacher


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_named_observation_order(self):
        fields = [torch.full((2, dim), float(i)) for i, dim in enumerate([29,29,3,6,3,3,3,29,29,29,29,29])]
        ref, prop, teacher = observations(*fields)
        self.assertEqual(ref.shape, (2,67))
        self.assertEqual(prop.shape, (2,96))
        self.assertEqual(teacher.shape, (2,160))
        torch.testing.assert_close(teacher[:, :67], ref)
        torch.testing.assert_close(teacher[:, 67:], prop[:, 3:])
        torch.testing.assert_close(prop[:, 9:38], fields[7] - fields[9])
        fields[-1].fill_(float("nan"))
        with self.assertRaises(ValueError):
            observations(*fields)

    def test_cvae_shapes_gradients_and_reference_free_decoder(self):
        model = CVAE(ModelConfig(hidden=(24, 16)))
        ref, prop = torch.randn(5,67), torch.randn(5,96)
        action, mu, logvar, z = model(ref, prop)
        self.assertEqual(action.shape, (5,29))
        self.assertEqual(z.shape, (5,32))
        self.assertFalse(torch.equal(z, mu))
        loss, _, _ = vae_loss(action, torch.randn_like(action), mu, logvar)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        deterministic = model(ref, prop, sample=False)
        torch.testing.assert_close(deterministic[3], deterministic[1])
        decoder = DecoderOnly(model).eval()
        traced = torch.jit.trace(decoder, (z.detach(), prop))
        torch.testing.assert_close(traced(z[:1].detach(), prop[:1]), model.decode(z[:1].detach(), prop[:1]))
        self.assertFalse(any("encoder" in k or "reference" in k for k in decoder.state_dict()))

    def test_kl_latent_sum_factor_and_zero(self):
        pred, label = torch.zeros(2,29), torch.zeros(2,29)
        mu, logvar = torch.ones(2,32), torch.zeros(2,32)
        loss, rec, kl = vae_loss(pred,label,mu,logvar)
        self.assertAlmostEqual(kl.item(),16)
        self.assertAlmostEqual(loss.item(),.16, places=6)
        self.assertAlmostEqual(vae_loss(pred,label,mu,logvar,kl_reduction="mean")[2].item(),.5)
        self.assertEqual(vae_loss(pred,label,mu*0,logvar)[2].item(),0)

    def test_dagger_teacher_sees_previous_student_action(self):
        seen = {}
        previous = torch.full((2,29),3.)
        snapshot = {"reference":torch.zeros(2,67), "proprio":torch.cat((torch.zeros(2,67),previous),-1),
                    "teacher_observation":torch.cat((torch.zeros(2,131),previous),-1)}

        def teacher(obs):
            seen["teacher_previous"] = obs[:, -29:].clone()
            return torch.ones(2,29)*11

        def student(ref,prop,sample):
            seen["student_previous"] = prop[:,-29:].clone()
            latent = torch.randn(2,32) if sample else torch.zeros(2,32)
            return torch.ones(2,29)*7, torch.zeros(2,32), torch.zeros(2,32), latent

        label,executed,fields = label_and_action(snapshot,teacher,student,sample=True)
        torch.testing.assert_close(seen["teacher_previous"],previous)
        torch.testing.assert_close(seen["student_previous"],previous)
        self.assertTrue((label == 11).all() and (executed == 7).all())
        self.assertFalse(torch.equal(fields["latent_used"],fields["mu"]))
        torch.testing.assert_close(fields["vae_action_clean"],executed)

    def test_normalizer_streaming_and_frozen(self):
        x = torch.randn(41,67)
        norm = FrozenStandardizer(67)
        norm.fit((x[:5], x[5:23], x[23:]))
        torch.testing.assert_close(norm.mean, x.mean(0))
        torch.testing.assert_close(norm.std, x.std(0, unbiased=False))
        before = copy.deepcopy(norm.state_dict())
        norm.train()
        norm(torch.randn(3,67)*100)
        for key in before:
            torch.testing.assert_close(before[key], norm.state_dict()[key])

    def test_teacher_exact_rsl_normalization_once(self):
        from rsl_rl.modules import ActorCritic, EmpiricalNormalization
        actor = ActorCritic(160, 160, 29, actor_hidden_dims=[32,16], critic_hidden_dims=[16], activation="elu")
        norm = EmpiricalNormalization(160)
        norm(torch.randn(43,160)*2+3)
        norm.eval()
        cfg = {"policy": {"actor_hidden_dims": [32,16], "activation": "elu", "class_name": "ActorCritic"}, "empirical_normalization": True, "clip_actions": None}
        with tempfile.TemporaryDirectory() as tmp:
            pt, yml = Path(tmp)/"unit.pt", Path(tmp)/"agent.yaml"
            torch.save({"model_state_dict":actor.state_dict(), "obs_norm_state_dict":norm.state_dict()}, pt)
            yml.write_text(yaml.safe_dump(cfg))
            teacher = Teacher(pt,yml)
            obs = torch.randn(7,160)
            with torch.no_grad():
                torch.testing.assert_close(teacher(obs), actor.act_inference(norm(obs)), rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            wrong = torch.load(pt,weights_only=False)
            del wrong["obs_norm_state_dict"]
            torch.save(wrong,pt)
            with self.assertRaises(ValueError):
                Teacher(pt,yml)

    @staticmethod
    def write_run(root, split, seed, round_, value=1):
        meta = {"split": split,"motion_id":"walk","collector_seed":seed,"round":round_,
                "mapping_hash":"mapping","motion_frames":30,"teacher_qualified":True}
        writer = ShardWriter(root,meta,shard_rows=4)
        writer.append({"reference":np.full((7,67),value,np.float32), "proprio":np.full((7,96),value*2,np.float32),
                       "teacher_label":np.full((7,29),value*3,np.float32), "reference_frame":np.arange(7,dtype=np.int64)})
        writer.close()

    def test_shards_mmap_integrity_partition_and_d0_only_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            d0, val, d1 = [Path(tmp)/n for n in ("d0","val","d1")]
            self.write_run(d0,"train",100,0,1)
            self.write_run(val,"val",101,0,1000)
            self.write_run(d1,"train",102,1,100)
            data = RolloutDataset([d0,val,d1],"train",cache_shards=1)
            self.assertEqual(len(data),14)
            self.assertEqual(len(RolloutDataset([d0,val,d1],"val")),7)
            self.assertEqual(data[0][0][0],1)
            self.assertEqual(data[13][0][0],100)
            norm = FrozenStandardizer(67)
            norm.fit(data.d0_batches("reference"))
            torch.testing.assert_close(norm.mean,torch.ones(67))
            self.assertIsInstance(mmap_npz(data.shards[0],"reference"),np.memmap)
            sampler = data.sampler(10000, torch.Generator().manual_seed(4))
            indices = np.array(list(sampler))
            self.assertLess(abs((indices<7).mean()-.25),.03)
            with self.assertRaises(FileExistsError):
                ShardWriter(d0,{"split":"train"})
            with open(data.shards[0],"ab") as stream:
                stream.write(b"corrupted")
            with self.assertRaises(ValueError):
                RolloutDataset([d0],"train")

    def test_incomplete_and_duplicate_seed_partitions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"one", Path(tmp)/"two"
            writer = ShardWriter(first,{"split":"train"})
            with self.assertRaises(ValueError):
                RolloutDataset([first],"train")
            # Different splits cannot share a collector seed for same motion/round.
            first = Path(tmp)/"real"
            self.write_run(first,"train",10,0)
            self.write_run(second,"val",10,0)
            with self.assertRaises(ValueError):
                RolloutDataset([first,second],"train")

    def test_partial_accumulation_matches_large_batch(self):
        sys.path.insert(0,str(ROOT/"scripts/stage2"))
        spec = importlib.util.spec_from_file_location("stage2_train",ROOT/"scripts/stage2/train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        a = CVAE(ModelConfig(hidden=(8,),grad_clip=100))
        b = copy.deepcopy(a)
        ref,prop,label = torch.randn(9,67),torch.randn(9,96),torch.randn(9,29)
        oa,ob = torch.optim.SGD(a.parameters(),lr=.01),torch.optim.SGD(b.parameters(),lr=.01)
        pred,mu,lv,_ = a(ref,prop,sample=False)
        vae_loss(pred,label,mu,lv)[0].backward()
        oa.step()
        for left,right in ((0,4),(4,7),(7,9)):
            pred,mu,lv,_ = b(ref[left:right],prop[left:right],sample=False)
            (vae_loss(pred,label[left:right],mu,lv)[0]*(right-left)).backward()
        module.optimizer_step(b,ob,9)
        for pa,pb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(pa,pb,rtol=1e-5,atol=1e-7)

    def test_checkpoint_contract_and_decoder_reload(self):
        model = CVAE(ModelConfig(hidden=(8,)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"unit.pt"
            torch.save({"config":asdict(model.config),"contract_hash":CONTRACT_HASH,"model":model.state_dict()},path)
            loaded,_ = load_student(path)
            ref,prop = torch.randn(2,67),torch.randn(2,96)
            torch.testing.assert_close(model(ref,prop,False)[0],loaded(ref,prop,False)[0])
            torch.save({"contract_hash":"wrong"},path)
            with self.assertRaises(ValueError):
                load_student(path)

    def test_training_resume_rng_and_new_round_frozen_stats(self):
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/"scripts/stage2"))
        spec = importlib.util.spec_from_file_location("stage2_train_resume",ROOT/"scripts/stage2/train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            roots = [Path(tmp)/n for n in ("d0","val","d1","val1")]
            self.write_run(roots[0],"train",100,0,1)
            self.write_run(roots[1],"val",101,0,2)
            self.write_run(roots[2],"train",100,1,10)
            self.write_run(roots[3],"val",101,1,20)
            full, first, resumed, aggregated = [Path(tmp)/n for n in ("full","first","resumed","aggregated")]

            def run(output, epochs, resume=None, new_round=False):
                data = roots if new_round else roots[:2]
                argv = ["train.py","--data",*[str(p) for p in data],"--output",str(output),
                        "--epochs",str(epochs),"--samples_per_epoch","9","--batch_size","4","--device","cpu"]
                if resume:
                    argv += ["--resume",str(resume)]
                if new_round:
                    argv += ["--new_round"]
                with patch.object(sys,"argv",argv), patch.object(module,"CVAE",side_effect=lambda: CVAE(ModelConfig(hidden=(8,)))):
                    module.main()

            run(full,3)
            run(first,2)
            run(resumed,3,first/"last.pt")
            reference = torch.load(full/"last.pt",weights_only=False)
            resumed_state = torch.load(resumed/"last.pt",weights_only=False)
            for key,value in reference["model"].items():
                torch.testing.assert_close(value,resumed_state["model"][key],rtol=0,atol=0)
            self.assertEqual(reference["optimizer_step"],3)
            run(aggregated,1,resumed/"last.pt",new_round=True)
            final = torch.load(aggregated/"last.pt",weights_only=False)
            for key in ("reference_norm.mean","reference_norm.std","proprio_norm.mean","proprio_norm.std"):
                torch.testing.assert_close(final["model"][key],reference["model"][key],rtol=0,atol=0)
            self.assertEqual(final["normalizer_d0_source"],reference["normalizer_d0_source"])


if __name__ == "__main__":
    unittest.main()
