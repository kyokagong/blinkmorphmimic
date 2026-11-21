"""Fine-tuning keeps parent identity strict and samples exact recovery poses."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import torch


class RecoveryTrainingTests(unittest.TestCase):
    def test_parent_receipt_requires_exact_checkpoint_and_full_optimizer_state(self):
        from morphmimic.train.recovery import verify_finetune_parent
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);checkpoint=p/'q1'/'ckpt_5000.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'parent')
            windows=p/'windows.json';windows.write_text('{}')
            digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
            receipt={'schema_version':1,'parent_iteration':5000,'target_iteration':6000,
                     'prepared_manifest_sha256':'a'*64,'windows_sha256':digest(windows),
                     'checkpoints':{'q1':{'path':str(checkpoint),'sha256':digest(checkpoint),'freeze_receipt':'parent-freeze.json'}}}
            path=p/'parent.json';path.write_text(json.dumps(receipt))
            cfg=SimpleNamespace(finetune_parent_receipt=str(path),robot='q1',iterations=6000,
                                recovery_windows=str(windows),recovery_windows_sha256=digest(windows),
                                finetune_parent_receipt_sha256=digest(path))
            called=[];identity=SimpleNamespace(sha256='a'*64,verify_checkpoint=lambda state,receipt:called.append(receipt))
            saved={'iteration':5000,'cfg':{'robot':'q1','optimizer':'muon'},'optim':{k:{'state':{0:{'step':1}}} for k in ('wm','policy','critic','kp','critic_kp')},'ema_policy':{'weight':torch.ones(1)},'ema_updates':1,'optimization_state':{'optimizer_steps':{'world_model':1,'motion':1,'human':1}}}
            result=verify_finetune_parent(saved,checkpoint,cfg,identity)
            self.assertEqual(result['iteration'],5000);self.assertTrue(result['optimizer_state_restored'])
            self.assertEqual(called,['parent-freeze.json'])
            for change in ({'iteration':4999},{'optim':{}},{'ema_policy':None},{'ema_updates':0},{'optimization_state':{'optimizer_steps':{}}}):
                with self.assertRaises(ValueError):verify_finetune_parent(dict(saved,**change),checkpoint,cfg,identity)
            saved['iteration']=6000
            saved['finetuned_from']={'iteration':5000,'sha256':'old-parent'}
            receipt.update(parent_iteration=6000,target_iteration=7000)
            path.write_text(json.dumps(receipt));cfg.iterations=7000;cfg.finetune_parent_receipt_sha256=digest(path)
            result=verify_finetune_parent(saved,checkpoint,cfg,identity)
            self.assertEqual(result['iteration'],6000)
            self.assertEqual(result['ancestor'],saved['finetuned_from'])
            saved['iteration']=5000
            with self.assertRaises(ValueError):verify_finetune_parent(saved,checkpoint,cfg,identity)
            saved['iteration']=6000
            original=path.read_text();path.write_text(original+' ')
            with self.assertRaises(ValueError):verify_finetune_parent(saved,checkpoint,cfg,identity)
            path.write_text(original)
            checkpoint.write_bytes(b'replaced')
            with self.assertRaises(ValueError):verify_finetune_parent(saved,checkpoint,cfg,identity)

    def test_exact_window_focus_is_category_balanced_and_preserves_base_sampling(self):
        from morphmimic.train.recovery import recovery_tensors,sample_recovery_frames
        rows=[dict(robot='q1',clip='getup',source_id='lafan/getup',category=c,start_frame=a,stop_frame=b,
                   stand_frame=180,hold_end_frame=240) for c,a,b in [('lying',10,20),('seated',60,80),('half_squat',120,125)]]
        doc={'schema_version':1,'fps':50,'training_manifest_sha256':'a'*64,'subset_sha256':{'q1':'b'*64},'windows':rows}
        lib=SimpleNamespace(clip_names=['getup'],clip_start=torch.tensor([0]),clip_len=torch.tensor([300]),device='cpu',fps=50)
        tensors=recovery_tensors(doc,lib,'q1','b'*64,'a'*64)
        command=SimpleNamespace(recovery=tensors,recovery_rho=.8)
        torch.manual_seed(123);base=torch.full((100000,),290,dtype=torch.long);frames=sample_recovery_frames(command,base)
        base_share=(frames==290).float().mean().item();self.assertAlmostEqual(base_share,.2,delta=.01)
        for a,b in ((10,20),(60,80),(120,125)):
            self.assertAlmostEqual(((frames>=a)&(frames<b)).float().mean().item(),.8/3,delta=.01)
        self.assertTrue(torch.all((frames==290)|((frames>=10)&(frames<20))|((frames>=60)&(frames<80))|((frames>=120)&(frames<125))))
        self.assertIs(sample_recovery_frames(SimpleNamespace(),base),base)
        for bad in (dict(doc,training_manifest_sha256='c'*64),dict(doc,windows=rows[:2]),dict(doc,windows=[dict(r,stop_frame=301) for r in rows])):
            with self.assertRaises(ValueError):recovery_tensors(bad,lib,'q1','b'*64,'a'*64)

