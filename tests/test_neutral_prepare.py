import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('prepare',Path(__file__).resolve().parents[1]/'scripts/prepare_unified_neutral.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class NeutralPrepareTests(unittest.TestCase):
    def test_inverse_source_windows_preserve_middle_and_exclusive_end(self):
        c=dict(original_source_frames=200,source_ramp_frames=25,source_start_frame=125)
        self.assertEqual(m.remap_index(c,0),125)
        self.assertEqual(m.remap_index(c,50),200)
        self.assertEqual(m.remap_index(c,200),375)
        with self.assertRaises(ValueError):m.remap_index(c,201)

    def test_recovery_labels_follow_retiming_including_in_core_hold(self):
        c=dict(original_source_frames=200,source_ramp_frames=25,source_start_frame=125)
        old=dict(windows=[dict(robot='q1',clip='getup',start_frame=0,stop_frame=20,stand_frame=100,hold_end_frame=200)],
            preview_poses=dict(lying=dict(q1=dict(clip='getup',start_frame=10,hold_frame=100))))
        rows=[dict(robot='q1',name='getup',neutral_contract=c)]
        new=m.remap_windows(old,rows)
        self.assertEqual(new['windows'][0]['stand_frame'],250)
        self.assertEqual(new['windows'][0]['hold_end_frame'],375)
        self.assertEqual(old['windows'][0]['start_frame'],0)
        self.assertEqual(new['preview_poses']['lying']['q1']['hold_frame'],250)

if __name__=='__main__':unittest.main()

class GeneratedNeutralInventoryTests(unittest.TestCase):
    @unittest.skipUnless((Path('/data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2')/'manifest.json').exists(), 'Unified corpus not generated')
    def test_full_rewrite_preserves_identity_and_has_self_contained_neutral_contracts(self):
        import json
        root=Path('/data/kk/blinkmorphmimic/prepared/inplace-recovery-neutral-20261117-v2')
        document=json.loads((root/'manifest.json').read_text())
        parent=json.loads(Path(document['derivation']['parent_manifest']['path']).read_text())
        excluded={(r['robot'],r['name']) for r in document['derivation']['excluded_recovery_references']}
        expected=[r for r in parent['clips'] if (r['robot'],r['name']) not in excluded]
        self.assertEqual(len(document['clips']),34621)
        self.assertEqual([(r['robot'],r['name'],r['source_id'],r['split']) for r in document['clips']],
            [(r['robot'],r['name'],r['source_id'],r['split']) for r in expected])
        for r in document['clips']:
            self.assertTrue(Path(r['output']).is_relative_to(root))
            c=r['neutral_contract']
            self.assertEqual(c['frames'],r['frames'])
            self.assertFalse(c['dynamics_qualified'])
            self.assertTrue(c['quality']['valid'])
            self.assertGreaterEqual(c['entry_hold_end_frame'],50)
            self.assertLessEqual(c['quality']['bridge_joint_convex_hull_error_rad'],2e-6)
            if r['dataset']=='motiondecode':self.assertFalse(r.get('human_available'))
