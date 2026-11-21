import unittest

class WorkerTests(unittest.TestCase):
    def test_budget_includes_startup_and_requires_real_training_time(self):
        from morphmimic.train.probe_worker import budget_remaining
        self.assertEqual(budget_remaining(100.,4.,.05),1.)
        self.assertEqual(budget_remaining(0.,0.,.05),0.)
        self.assertEqual(budget_remaining(100.,6.,.05),0.)
        with self.assertRaises(ValueError):budget_remaining(10.,-1.,.05)
        with self.assertRaises(ValueError):budget_remaining(10.,1.,.051)

    def test_combined_failures_keep_interface_and_declared_weights(self):
        from morphmimic.train.probe_worker import aggregate_failures
        records={'motion':[{'clip':'a'}],'human-five':[{'clip':'a'}],'human-full':[]}
        combined=aggregate_failures(records,{'motion':.5,'human-five':.3,'human-full':.2})
        self.assertEqual([(x['interface'],x['weight']) for x in combined],[('motion',.5),('human-five',.3)])
        with self.assertRaises(ValueError):aggregate_failures(records,{'motion':1.,'human-five':-1.,'human-full':1.})

    def test_fixed_latest_checkpoint_is_numbered_and_stable(self):
        from morphmimic.train.probe_worker import latest_completed
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'ckpt_1000.pt').write_bytes(b'a');(p/'ckpt_2000.pt').write_bytes(b'b')
            (p/'latest.pt').symlink_to('ckpt_2000.pt');(p/'ckpt_3000.pt').symlink_to('ckpt_2000.pt')
            self.assertEqual(latest_completed(p).name,'ckpt_2000.pt')

class ClipSelectionTests(unittest.TestCase):
    def test_clip_selection_rejects_duplicates_out_of_bounds_and_empty(self):
        from morphmimic.train.probe_worker import validate_clip_ids
        self.assertEqual(validate_clip_ids([2,0],3),[2,0])
        for bad in ([],[1,1],[-1],[3],[True]):
            with self.assertRaises(ValueError):validate_clip_ids(bad,3)

class TrainingProofTests(unittest.TestCase):
    def test_manifest_accepts_only_exact_training_paths_and_locks_hashes(self):
        from morphmimic.train.probe_worker import validate_training_rows
        rows=[{'robot':'q1','name':'a','split':'train','output':'/one/q1/a.npz','source_id':'seed/a','dataset':'seed','human_available':True,'frames':5}]
        selected=validate_training_rows(rows,'q1',['/one/q1/a.npz'])
        self.assertEqual(selected[0]['source_id'],'seed/a')
        for bad in (['/other/q1/a.npz'],['/one/q1/a.npz','/one/q1/a.npz']):
            with self.assertRaises(ValueError):validate_training_rows(rows,'q1',bad)
        rows[0]['split']='val'
        with self.assertRaises(ValueError):validate_training_rows(rows,'q1',['/one/q1/a.npz'])

class WeightValidationTests(unittest.TestCase):
    def test_rejects_overflowing_weight_total(self):
        from morphmimic.train.probe_worker import aggregate_failures
        records={'motion':[],'human-five':[],'human-full':[]}
        with self.assertRaises(ValueError):aggregate_failures(records,{'motion':1e308,'human-five':1e308,'human-full':0.})

class WatcherIterationTests(unittest.TestCase):
    def test_checkpoint_zero_is_new_only_until_completed(self):
        from morphmimic.train.probe_worker import needs_probe
        self.assertTrue(needs_probe({},'q1',0))
        self.assertFalse(needs_probe({'q1':0},'q1',0))
        self.assertTrue(needs_probe({'q1':0},'q1',1000))
        with self.assertRaises(ValueError):needs_probe({'q1':None},'q1',0)
        with self.assertRaises(ValueError):needs_probe({},'q1',-1)

class StratifiedPoolTests(unittest.TestCase):
    def test_all_three_corpora_fit_and_budget_is_strict(self):
        from scripts.probe_optimized import stratified_pool
        rows=[dict(name=f'{d}{i}',dataset=d,frames=n) for d,n in
              [('seed',10),('seed',100),('lafan1',20),('motiondecode',30)] for i in range(2)]
        chosen=stratified_pool(rows,3,57,13)
        self.assertEqual({r['dataset'] for r in chosen},{'seed','lafan1','motiondecode'})
        self.assertLessEqual(sum(r['frames']-1 for r in chosen),57)
        self.assertEqual(chosen,stratified_pool(rows,3,57,13))
        with self.assertRaises(ValueError):stratified_pool(rows,3,50,13)
        with self.assertRaises(ValueError):stratified_pool([r for r in rows if r['dataset']!='lafan1'],3,1000,13)

class PerInterfaceSelectionTests(unittest.TestCase):
    def test_feasible_selection_covers_corpora_and_reports_unfit_whole_clips(self):
        from scripts.probe_optimized import select_checkpoint_clips
        rows=[dict(name='s',dataset='seed',frames=6,human_available=True),
              dict(name='l',dataset='lafan1',frames=11,human_available=True),
              dict(name='m',dataset='motiondecode',frames=8,human_available=False),
              dict(name='long',dataset='seed',frames=101,human_available=True)]
        selected,details=select_checkpoint_clips(rows,'motion',4,30,1000)
        self.assertEqual({r['dataset'] for r in selected},{'seed','lafan1','motiondecode'})
        self.assertLessEqual(sum(r['frames']-1 for r in selected),30)
        self.assertEqual([r['name'] for r in details['over_budget_clips']],['long'])
        self.assertEqual(selected,select_checkpoint_clips(rows,'motion',4,30,1000)[0])
        human,_=select_checkpoint_clips(rows,'human-five',4,30,1000)
        self.assertEqual({r['dataset'] for r in human},{'seed','lafan1'})
        self.assertTrue(all(r['human_available'] for r in human))

    def test_tight_clip_limit_rotates_datasets_across_checkpoints(self):
        from scripts.probe_optimized import select_checkpoint_clips
        rows=[dict(name=d,dataset=d,frames=5,human_available=d!='motiondecode') for d in ('seed','lafan1','motiondecode')]
        observed=[]
        for iteration in (0,1000,2000):
            selected,_=select_checkpoint_clips(rows,'motion',1,10,iteration)
            observed.append(selected[0]['dataset'])
        self.assertEqual(set(observed),{'seed','lafan1','motiondecode'})
        with self.assertRaises(ValueError):select_checkpoint_clips(rows,'motion',1,1,0)

class ProbeWindowSelectionTests(unittest.TestCase):
    def test_bounded_windows_cover_sources_and_record_exclusions(self):
        from scripts.probe_optimized import select_checkpoint_clips, choose_window
        rows=[dict(name=d,dataset=d,frames=n,human_available=d!='motiondecode') for d,n in
              [('seed',300),('lafan1',10000),('motiondecode',1200)]]
        rows.append(dict(name='short',dataset='seed',frames=256,human_available=True))
        chosen,detail=select_checkpoint_clips(rows,'motion',4,768,1000,window_transitions=256)
        self.assertEqual({x['dataset'] for x in chosen},{'seed','lafan1','motiondecode'})
        self.assertEqual([x['name'] for x in detail['ineligible_short_clips']],['short'])
        for row in chosen:
            start,end=choose_window(row,'motion',1000,256)
            self.assertEqual(end-start,256)
            self.assertLess(end,row['frames'])
            self.assertEqual((start,end),choose_window(row,'motion',1000,256))
        human,_=select_checkpoint_clips(rows,'human-five',4,768,1000,window_transitions=256)
        self.assertTrue(all(x['human_available'] for x in human))
        with self.assertRaises(ValueError):select_checkpoint_clips(rows,'motion',4,255,0,window_transitions=256)
