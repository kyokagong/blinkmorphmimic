import unittest
from scripts.prepare_optimized import composite_rows


def row(robot, dataset, name, group, split='train'):
    return dict(robot=robot, dataset=dataset, name=name, group_id=group, split=split,
                output=f'{robot}/{name}.npz', output_sha256='a'*64,
                human_sidecar=f'human50/{robot}/{name}.npz', human_sha256='b'*64)


class CompositeTests(unittest.TestCase):
    def test_preserves_motion_only_and_triplets(self):
        h=[row(r,'seed','a','seed/a') for r in ('q1','bumi','g1')]
        m=[row(r,'motiondecode','b','motiondecode/b') for r in ('q1','bumi','g1')]
        rows, errors=composite_rows({'complete':True,'errors':[],'clips':h},
            {'source_fps':120,'source_fps_requires_confirmation':False,
             'source_fps_confirmation':{'value':120,'by':'user'},'errors':[], 'clips':m}, '/human','/motion')
        self.assertEqual(len(rows),6)
        self.assertEqual([r['human_available'] for r in rows].count(False),3)
        self.assertEqual(errors,[])

    def test_rejects_group_leak_and_unconfirmed_timing(self):
        h=[row('q1','seed','a','seed/a'),row('g1','seed','b','seed/a','val')]
        with self.assertRaisesRegex(ValueError,'split'):
            composite_rows({'complete':True,'errors':[],'clips':h},
                {'source_fps':120,'source_fps_requires_confirmation':False,'source_fps_confirmation':{'value':120,'by':'user'},'clips':[]},'/human','/motion')
        with self.assertRaisesRegex(ValueError,'fps'):
            composite_rows({'complete':True,'errors':[],'clips':[]},
                {'source_fps':120,'source_fps_requires_confirmation':True,'clips':[]},'/human','/motion')

if __name__=='__main__': unittest.main()

class PresetTests(unittest.TestCase):
    def test_fixed_joint_preset(self):
        from scripts.train_optimized import config_values
        cfg=config_values('/data/composite','/data/human50','/run','joint',[0,2,3])
        self.assertEqual(cfg['num_envs'],2048)
        self.assertEqual(cfg['iterations'],5000)
        self.assertEqual(cfg['ckpt_every'],1000)
        self.assertEqual(cfg['optimizer'],'muon')
        self.assertEqual(cfg['kp_fixed_horizon'],1)
        self.assertEqual(cfg['robots'],('q1','bumi','g1'))
        self.assertTrue(cfg['kp_interface'])
        self.assertEqual(cfg['human_root'],'/data/human50')

class EvaluationTests(unittest.TestCase):
    def test_cases_separate_interfaces_and_dataset(self):
        from scripts.evaluate_optimized import cases
        rows=[dict(robot='q1',dataset='seed',split='val',human_available=True),
              dict(robot='q1',dataset='motiondecode',split='val',human_available=False)]
        found=cases(rows,'q1')
        self.assertEqual(set(found),{('seed','motion'),('seed','five'),('seed','full'),('motiondecode','motion')})
        self.assertEqual(len(found['seed','five']),1)

    def test_aggregation_reports_missing_checkpoint_cases(self):
        from scripts.evaluate_optimized import audit_results
        report=audit_results([dict(iteration=1000,robot='q1',dataset='seed',interface='five',status='ok')],
                             [(1000,'q1','seed','five'),(2000,'q1','seed','five')])
        self.assertEqual(report[1]['status'],'missing')

class PlotTests(unittest.TestCase):
    def test_failure_creates_gap_in_curve(self):
        from scripts.plot_optimized import series
        rows=[dict(iteration=1000,status='ok',mean_norm_return=.5),
              dict(iteration=2000,status='error'),dict(iteration=3000,status='ok',mean_norm_return=.7)]
        x,y=series(rows,'mean_norm_return')
        self.assertEqual(x,[1000,2000,3000])
        self.assertNotEqual(y[1],y[1])

class SourceIdentityTests(unittest.TestCase):
    def test_robot_peers_cannot_disagree_on_source_group(self):
        rows=[row('q1','seed','a','seed/a'),row('g1','seed','a','seed/b')]
        with self.assertRaisesRegex(ValueError,'peer source'):
            composite_rows({'complete':True,'clips':rows},
                {'source_fps':120,'source_fps_requires_confirmation':False,'source_fps_confirmation':{'value':120,'by':'user'},'clips':[]},'/human','/motion')

class SidecarSafetyTests(unittest.TestCase):
    def test_rejects_human_sidecar_for_motion_only_clip(self):
        import tempfile
        from pathlib import Path
        from scripts.prepare_optimized import check_no_motion_sidecar
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'human50/q1/a.npz';path.parent.mkdir(parents=True);path.write_bytes(b'not a human pair')
            with self.assertRaisesRegex(ValueError,'sidecar'):
                check_no_motion_sidecar(root,dict(robot='q1',name='a'))

class EvaluationAuditTests(unittest.TestCase):
    def test_unexpected_cases_are_not_silently_dropped(self):
        from scripts.evaluate_optimized import audit_results
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            audit_results([dict(iteration=1000,robot='q1',dataset='seed',interface='five',status='error')],[])

class CapacityTests(unittest.TestCase):
    def test_preset_uses_explicit_memory_and_data_contracts(self):
        from scripts.train_optimized import config_values
        c=config_values('/data/composite','/data/human50','/run','joint',[0,2,3])
        self.assertEqual(c['replay_capacity'],256)
        self.assertEqual(c['wm_batch'],1024)
        self.assertEqual(c['wm_batches'],5)
        self.assertEqual(c['wm_hidden'],(1024,1024,1024))
        self.assertEqual(c['policy_hidden'],(2048,1024,512,256))
        self.assertTrue(c['allow_missing_human'])
        self.assertEqual(c['prepared_manifest'],'/data/composite/manifest.json')
        self.assertEqual(c['compile'],'default')

class PipelineCorrectionsTests(unittest.TestCase):
    def test_zero_checkpoint_is_part_of_audit_without_fabrication(self):
        from scripts.evaluate_optimized import checkpoint_iterations,audit_results
        self.assertEqual(list(checkpoint_iterations()),list(range(0,5001,1000)))
        expected=[(0,'q1','seed','motion')]
        self.assertEqual(audit_results([],expected)[0]['status'],'missing')

    def test_drift_values_are_flattened_for_curves(self):
        from scripts.evaluate_optimized import flatten_summary
        result=flatten_summary({'mean_norm_return':.9,'drift':{'mean_xy':.1,'final_xy':.2,'mean_abs_yaw_deg':3.}})
        self.assertEqual((result['drift_mean_xy'],result['drift_final_xy'],result['drift_mean_abs_yaw_deg']),(.1,.2,3.))
        self.assertEqual(result['drift']['mean_xy'],.1)

    def test_preset_enables_physical_reward_terms(self):
        from scripts.train_optimized import config_values
        cfg=config_values('/data','/human','/run','joint',[0,2,3])
        self.assertEqual(cfg['contact_agreement_weight'],.1)
        self.assertEqual(cfg['foot_slip_weight'],-.001)

    def test_cli_launches_and_passes_resume(self):
        import json,sys,tempfile,types
        from pathlib import Path
        from unittest.mock import patch
        import scripts.train_optimized as script
        calls=[]
        class Config:
            def __init__(self,**values):self.values=values
        trainer=types.ModuleType('morphmimic.train.trainer');trainer.TrainCfg=Config
        guard=types.ModuleType('morphmimic.scripts.gpu_guard');guard.select_gpus=lambda ids,allowed:calls.append(('gpu',ids))
        launch=types.ModuleType('morphmimic.scripts.train');launch.launch=lambda cfg,resume:calls.append(('launch',cfg.values,resume))
        with tempfile.TemporaryDirectory() as d:
            Path(d,'manifest.json').write_text(json.dumps({'complete':True,'human_root':'/human'}))
            argv=['probe','--data-root',d,'--log-dir',d,'--run-name','r','--gpu-ids','0','2','3','--resume','/run/ckpt']
            with patch.dict(sys.modules,{'morphmimic.train.trainer':trainer,'morphmimic.scripts.gpu_guard':guard,'morphmimic.scripts.train':launch}):
                with patch.object(sys,'argv',argv):script.main()
        self.assertEqual(calls[0],('gpu',[0,2,3]))
        self.assertEqual(calls[1][0],'launch');self.assertEqual(calls[1][2],'/run/ckpt')
        self.assertEqual(calls[1][1]['num_envs'],2048)

class PlotArtifactTests(unittest.TestCase):
    def test_real_drift_figures_include_missing_zero_as_gap(self):
        import importlib.util
        if importlib.util.find_spec('matplotlib') is None or importlib.util.find_spec('tensorboard') is None:
            self.skipTest('matplotlib and tensorboard required')
        import json,sys,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from torch.utils.tensorboard import SummaryWriter
        from scripts.plot_optimized import main
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();run=root/'run';run.mkdir()
            (data/'manifest.json').write_text(json.dumps({'clips':[dict(robot=r,dataset='seed',split='val',human_available=False) for r in ('q1','bumi','g1')]}))
            from scripts.prepare_optimized import sha
            manifest_hash=sha(data/'manifest.json')
            rows=[]
            for robot in ('q1','bumi','g1'):
                (run/robot).mkdir();writer=SummaryWriter(str(run/robot));writer.add_scalar('loss',.5,1);writer.add_scalar('loss',float('nan'),1000);writer.add_scalar('loss',.75,2000);writer.close()
                for it in range(1000,10001,1000):
                    ckpt=run/robot/f'ckpt_{it}.pt';ckpt.write_bytes(f'{robot}-{it}'.encode())
                    stem=f'{robot}_seed_motion_{it}'
                    csv=root/(stem+'.csv');csv.write_text('name\na\n')
                    subset=root/(stem+'.yaml');subset.write_text('{}')
                    rows.append(dict(iteration=it,robot=robot,dataset='seed',interface='motion',status='ok',
                        manifest_sha256=manifest_hash,metric_definition='legacy_tracking_reward_v1',normalization_r_max=.1,
                        checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),per_clip_sha256=sha(csv),subset_sha256=sha(subset),
                        mean_norm_return=.9,failure_rate=.1,drift={'mean_xy':.1,'final_xy':.2,'mean_abs_yaw_deg':3.}))
            evaluation=root/'eval.json';evaluation.write_text(json.dumps(rows));out=root/'plots'
            argv=['plot','--run-root',str(run),'--data-root',str(data),'--evaluation',str(evaluation),'--out',str(out),'--iterations',*[str(i) for i in range(0,10001,1000)]]
            with patch.object(sys,'argv',argv):
                with self.assertRaises(SystemExit):main()
            audit=json.loads((out/'evaluation-audit.json').read_text())
            self.assertEqual(sum(r['iteration']==0 and r['status']=='missing' for r in audit),3)
            scalar_text=(out/'training-scalars.json').read_text();self.assertNotIn('NaN',scalar_text)
            training=json.loads(scalar_text);self.assertEqual(training['q1']['tags']['loss'][1]['status'],'nonfinite')
            self.assertIsNone(training['q1']['tags']['loss'][1]['value'])
            checkpoint_rows=json.loads((out/'training-checkpoints.json').read_text())
            by_step={r['iteration']:r for r in checkpoint_rows if r['robot']=='q1' and r['tag']=='loss'}
            self.assertEqual(by_step[0]['status'],'missing');self.assertEqual(by_step[1000]['status'],'nonfinite')
            self.assertEqual((by_step[2000]['status'],by_step[2000]['value']),('ok',.75))
            import csv
            with (out/'training-checkpoints.csv').open(newline='') as stream:
                csv_rows=list(csv.DictReader(stream))
            self.assertEqual(len(csv_rows),len(checkpoint_rows))
            for metric in ('drift_mean_xy','drift_final_xy','drift_mean_abs_yaw_deg'):
                for suffix in ('png','svg'):self.assertGreater((out/f'eval_{metric}.{suffix}').stat().st_size,0)

class IncrementalEvaluationTests(unittest.TestCase):
    def test_selection_is_strict_and_preserves_default_full_sweep(self):
        from scripts.evaluate_optimized import validate_selection
        self.assertEqual(validate_selection(None,None),(list(range(0,5001,1000)),['q1','bumi','g1']))
        self.assertEqual(validate_selection([0,2000],['g1']),([0,2000],['g1']))
        for iterations,robots in [([0,0],['q1']),([1],['q1']),([-1000],['q1']),([11000],['q1']),([0],['bad']),([0],['q1','q1'])]:
            with self.assertRaises(ValueError):validate_selection(iterations,robots)

    def test_multiple_results_cannot_mix_data_or_checkpoint_hashes(self):
        import json,tempfile
        from pathlib import Path
        from scripts.plot_optimized import load_results
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=p/'a.json';b=p/'b.json';digest='a'*64
            base=dict(iteration=0,robot='q1',dataset='seed',interface='motion',status='error',manifest_sha256=digest)
            a.write_text(json.dumps([base]));b.write_text(json.dumps([dict(base,iteration=1000)]))
            self.assertEqual(len(load_results([a,b],digest)),2)
            b.write_text(json.dumps([dict(base,iteration=1000,manifest_sha256='b'*64)]))
            with self.assertRaisesRegex(ValueError,'manifest'):load_results([a,b],digest)

    def test_waiter_checks_all_numbered_files_and_rejects_symlink(self):
        import tempfile
        from pathlib import Path
        from scripts.watch_evaluations import ready_checkpoints
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for robot in ('q1','bumi','g1'):(root/robot).mkdir()
            self.assertIsNone(ready_checkpoints(root,0))
            for robot in ('q1','bumi','g1'):(root/robot/'ckpt_0.pt').write_bytes(robot.encode())
            found=ready_checkpoints(root,0)
            self.assertEqual(set(found),{'q1','bumi','g1'})
            (root/'g1/ckpt_0.pt').unlink();(root/'g1/ckpt_0.pt').symlink_to(root/'q1/ckpt_0.pt')
            with self.assertRaises(ValueError):ready_checkpoints(root,0)

class EvaluationIntegrityTests(unittest.TestCase):
    def test_conflicting_checkpoint_hash_or_changed_artifact_is_rejected(self):
        import json,tempfile
        from pathlib import Path
        from scripts.plot_optimized import load_results
        from scripts.prepare_optimized import sha
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ckpt=root/'ckpt_0.pt';ckpt.write_bytes(b'checkpoint')
            result=root/'results.json';csv=root/'q1_seed_motion_0.csv';csv.write_text('x')
            subset=root/'q1_seed_motion_0.yaml';subset.write_text('{}')
            row=dict(iteration=0,robot='q1',dataset='seed',interface='motion',status='ok',manifest_sha256='a'*64,
                     metric_definition='legacy_tracking_reward_v1',normalization_r_max=.1,checkpoint=str(ckpt),
                     checkpoint_sha256=sha(ckpt),per_clip_sha256=sha(csv),subset_sha256=sha(subset))
            result.write_text(json.dumps([row]));self.assertEqual(len(load_results([root],'a'*64)),1)
            csv.write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact'):load_results([root],'a'*64)
            csv.write_text('x');ckpt.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):load_results([root],'a'*64)

    def test_cli_rejects_duplicate_iterations_before_creating_output(self):
        import os,subprocess,sys,tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'out'
            cmd=[sys.executable,'-m','scripts.evaluate_optimized','--run-root',d,'--data-root',d,
                 '--out',str(out),'--iterations','0','0']
            result=subprocess.run(cmd,env=dict(os.environ,PYTHONPATH='src:.'),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0);self.assertIn('unique',result.stderr)
            self.assertFalse(out.exists())

class CheckpointScalarTests(unittest.TestCase):
    def test_exact_checkpoint_scalars_mark_missing_nonfinite_and_duplicates(self):
        from types import SimpleNamespace
        from scripts.plot_optimized import normalize_scalar_events,training_checkpoint_rows
        events=[SimpleNamespace(step=10,value=.5),SimpleNamespace(step=1000,value=float('nan')),
                SimpleNamespace(step=2000,value=.2),SimpleNamespace(step=2000,value=.3),
                SimpleNamespace(step=3000,value=.4)]
        points=normalize_scalar_events(events)
        self.assertEqual(points[1],dict(iteration=1000,value=None,status='nonfinite'))
        rows=training_checkpoint_rows({'q1':{'status':'ok','tags':{'loss':points}}})
        q1={r['iteration']:r for r in rows if r['robot']=='q1' and r['tag']=='loss'}
        self.assertEqual((q1[0]['status'],q1[0]['value']),('missing',None))
        self.assertEqual((q1[1000]['status'],q1[1000]['value']),('nonfinite',None))
        self.assertEqual((q1[2000]['status'],q1[2000]['value']),('ambiguous',None))
        self.assertEqual((q1[3000]['status'],q1[3000]['value']),('ok',.4))
        import json
        json.dumps(rows,allow_nan=False)
